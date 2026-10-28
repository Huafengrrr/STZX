package com.example.myapplication1

import android.Manifest
import android.content.Context
import android.location.Location
import android.location.LocationListener
import android.location.LocationManager
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.util.Log
import androidx.core.content.ContextCompat
import okhttp3.Call
import okhttp3.Callback
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import okhttp3.Response
import org.json.JSONObject
import java.io.IOException
import java.util.Locale
import java.util.Timer
import java.util.TimerTask

/**
 * 位置管理器
 * 负责后台定位巡航、坐标上传云端、高德逆地理编码、用户主动查询位置
 */
class LocationHelper(
    private val context: Context,
    private val client: OkHttpClient,
    private val uniCloudUrl: String,
    private val amapWebKey: String,
    private val getDeviceId: () -> String,
    private val onSilentAddressFetched: (currentStreet: String, intersectionAlert: String) -> Unit,
    private val onActiveAddressFetched: (fullText: String) -> Unit,
    private val onNavProgressCheck: () -> Unit
) {
    companion object {
        private const val TAG = "Location"
    }

    // 全局坐标缓存，供跌倒等紧急情况瞬间提取最后一次已知位置
    var lastKnownLon: Double = 0.0
    var lastKnownLat: Double = 0.0

    private var locationTimer: Timer? = null
    var lastAnnouncedStreet = ""
    var lastIntersectionTime = 0L

    /**
     * 启动后台位置巡航定时器，每 15 秒静默刷新一次位置
     */
    fun startLocationCruise() {
        locationTimer = Timer()
        locationTimer?.scheduleAtFixedRate(object : TimerTask() {
            override fun run() {
                silentlyFetchLocation()
            }
        }, 5000, 15000)
    }

    /**
     * 停止后台定位巡航，释放资源
     */
    fun stopLocationCruise() {
        locationTimer?.cancel()
        locationTimer = null
    }

    /**
     * 后台静默抓取一次最新位置并上传云端
     */
    fun silentlyFetchLocation() {
        if (ContextCompat.checkSelfPermission(context, Manifest.permission.ACCESS_FINE_LOCATION)
            != android.content.pm.PackageManager.PERMISSION_GRANTED) return

        val locationManager = context.getSystemService(Context.LOCATION_SERVICE) as LocationManager
        val provider = if (locationManager.isProviderEnabled(LocationManager.GPS_PROVIDER)) {
            LocationManager.GPS_PROVIDER
        } else {
            LocationManager.NETWORK_PROVIDER
        }

        // 先用最后已知位置立即上传，防止 onLocationChanged 永远不触发
        try {
            val lastLoc = locationManager.getLastKnownLocation(provider)
            if (lastLoc != null) {
                val (gcjLon, gcjLat) = StereoVision.wgs84ToGcj02(lastLoc.longitude, lastLoc.latitude)
                lastKnownLon = gcjLon
                lastKnownLat = gcjLat
                Log.i(TAG, "使用最后已知位置上传：$gcjLon, $gcjLat")
                uploadLocation(gcjLon, gcjLat, "Normal")
            } else {
                Log.w(TAG, "getLastKnownLocation 返回 null，无历史位置可用")
            }
        } catch (e: Exception) {
            Log.e(TAG, "获取最后已知位置失败: ${e.message}")
        }

        // 向硬件强制请求一次最新鲜坐标
        try {
            locationManager.requestLocationUpdates(provider, 0L, 0f, object : LocationListener {
                override fun onLocationChanged(loc: Location) {
                    locationManager.removeUpdates(this)

                    val (gcjLon, gcjLat) = StereoVision.wgs84ToGcj02(loc.longitude, loc.latitude)
                    lastKnownLon = gcjLon
                    lastKnownLat = gcjLat

                    fetchAddressFromAmap(gcjLon, gcjLat, isSilent = true)
                    onNavProgressCheck()
                    uploadLocation(gcjLon, gcjLat, "Normal")
                }
                override fun onStatusChanged(provider: String?, status: Int, extras: Bundle?) {}
                override fun onProviderEnabled(provider: String) {}
                override fun onProviderDisabled(provider: String) {}
            }, Looper.getMainLooper())
        } catch (e: Exception) {
            Log.e(TAG, "后台强制定位失败", e)
        }
    }

    /**
     * 用户主动询问位置时调用，强制获取最新 GPS 坐标后逆解析播报
     */
    fun fetchCurrentLocationForUser() {
        if (ContextCompat.checkSelfPermission(context, Manifest.permission.ACCESS_FINE_LOCATION)
            != android.content.pm.PackageManager.PERMISSION_GRANTED) {
            onActiveAddressFetched("缺少定位权限，请在设置中开启")
            return
        }

        val locationManager = context.getSystemService(Context.LOCATION_SERVICE) as LocationManager
        val provider = if (locationManager.isProviderEnabled(LocationManager.GPS_PROVIDER)) {
            LocationManager.GPS_PROVIDER
        } else {
            LocationManager.NETWORK_PROVIDER
        }

        locationManager.requestLocationUpdates(provider, 0L, 0f, object : LocationListener {
            override fun onLocationChanged(loc: Location) {
                locationManager.removeUpdates(this)

                val (gcjLon, gcjLat) = StereoVision.wgs84ToGcj02(loc.longitude, loc.latitude)
                lastKnownLon = gcjLon
                lastKnownLat = gcjLat

                fetchAddressFromAmap(gcjLon, gcjLat, isSilent = false)
                uploadLocation(gcjLon, gcjLat, "Normal")
            }
            override fun onStatusChanged(provider: String?, status: Int, extras: Bundle?) {}
            override fun onProviderEnabled(provider: String) {}
            override fun onProviderDisabled(provider: String) {}
        })
    }

    /**
     * 访问高德逆地理编码 API，解析当前位置并分发播报
     * @param isSilent true 为后台巡航，只在关键节点发声；false 为用户主动询问，必须发声
     */
    fun fetchAddressFromAmap(lon: Double, lat: Double, isSilent: Boolean) {
        val formattedLon = String.format(Locale.US, "%.6f", lon)
        val formattedLat = String.format(Locale.US, "%.6f", lat)
        val url = "https://restapi.amap.com/v3/geocode/regeo?location=$formattedLon,$formattedLat&extensions=all&key=$amapWebKey"

        client.newCall(Request.Builder().url(url).get().build()).enqueue(object : Callback {
            override fun onFailure(call: Call, e: IOException) {
                if (!isSilent) onActiveAddressFetched("网络连接失败，无法查询位置")
            }

            override fun onResponse(call: Call, response: Response) {
                response.use {
                    if (!it.isSuccessful) {
                        if (!isSilent) onActiveAddressFetched("定位服务器返回错误")
                        return
                    }
                    try {
                        val json = JSONObject(it.body?.string() ?: "")
                        if (json.getString("status") != "1") {
                            if (!isSilent) onActiveAddressFetched("未能获取详细街道信息")
                            return
                        }

                        val regeocode = json.getJSONObject("regeocode")
                        val addressComponent = regeocode.getJSONObject("addressComponent")

                        // 1. 当前街道名
                        var currentStreet = ""
                        val streetNumber = addressComponent.optJSONObject("streetNumber")
                        if (streetNumber != null && streetNumber.has("street") && !streetNumber.isNull("street")) {
                            val streetStr = streetNumber.getString("street")
                            if (streetStr.isNotEmpty() && streetStr != "[]") {
                                currentStreet = streetStr
                            }
                        }

                        // 2. 完整地址
                        val fullAddress = regeocode.getString("formatted_address")

                        // 3. 交叉路口预警
                        var intersectionAlert = ""
                        try {
                            if (regeocode.has("roadinters")) {
                                val intersElement = regeocode.get("roadinters")
                                val intersArray = when {
                                    intersElement is org.json.JSONArray -> intersElement
                                    intersElement is String && intersElement.startsWith("[") -> org.json.JSONArray(intersElement)
                                    else -> null
                                }
                                if (intersArray != null && intersArray.length() > 0) {
                                    val nearestInter = intersArray.getJSONObject(0)
                                    val distToInter = nearestInter.optString("distance", "999").toDoubleOrNull() ?: 999.0
                                    if (distToInter < 60.0) {
                                        val road1 = nearestInter.optString("first_name", "")
                                        val road2 = nearestInter.optString("second_name", "")
                                        if (road1.isNotEmpty() && road2.isNotEmpty()) {
                                            intersectionAlert = "前方大约 ${distToInter.toInt()}米，到达 $road1 与 $road2 交叉路口。"
                                        }
                                    }
                                }
                            }
                        } catch (e: Exception) {
                            Log.e("AMAP", "路口解析异常", e)
                        }

                        // 4. 周边道路信息（仅用户主动询问时播报）
                        var roadsAlert = ""
                        try {
                            if (regeocode.has("roads")) {
                                val roadsElement = regeocode.get("roads")
                                val roadsArray = when {
                                    roadsElement is org.json.JSONArray -> roadsElement
                                    roadsElement is String && roadsElement.startsWith("[") -> org.json.JSONArray(roadsElement)
                                    else -> null
                                }
                                if (roadsArray != null && roadsArray.length() > 0) {
                                    val roadInfoList = mutableListOf<String>()
                                    val maxRoads = minOf(2, roadsArray.length())
                                    for (i in 0 until maxRoads) {
                                        val road = roadsArray.getJSONObject(i)
                                        val roadName = road.optString("name", "")
                                        val direction = road.optString("direction", "")
                                        if (roadName.isNotEmpty() && direction.isNotEmpty() && direction != "[]") {
                                            val dirText = if (direction.endsWith("侧") || direction.endsWith("边")) direction else "${direction}侧"
                                            roadInfoList.add("${dirText}是$roadName")
                                        }
                                    }
                                    if (roadInfoList.isNotEmpty()) {
                                        roadsAlert = "您的" + roadInfoList.joinToString("，") + "。"
                                    }
                                }
                            }
                        } catch (e: Exception) {
                            Log.e("AMAP", "周边道路解析异常", e)
                        }

                        // 5. 播报路由分发
                        if (!isSilent) {
                            val speakText = buildString {
                                append("您当前位于：$fullAddress。")
                                if (intersectionAlert.isNotEmpty()) append(intersectionAlert)
                                if (roadsAlert.isNotEmpty()) append(roadsAlert)
                            }
                            onActiveAddressFetched(speakText)
                        } else {
                            onSilentAddressFetched(currentStreet, intersectionAlert)
                        }

                    } catch (e: Exception) {
                        Log.e("AMAP", "高德解析错误", e)
                        if (!isSilent) onActiveAddressFetched("解析位置数据出错了")
                    }
                }
            }
        })
    }

    /**
     * 上传位置数据到 uniCloud 云函数
     * 含坐标有效性过滤、失败自动重试（最多 2 次）
     */
    fun uploadLocation(lon: Double, lat: Double, status: String = "Normal", retryCount: Int = 0) {
        if (lon == 0.0 && lat == 0.0) {
            Log.w(TAG, "坐标无效(0,0)，跳过上传")
            return
        }

        val deviceId = getDeviceId()
        val json = JSONObject().apply {
            put("action", "uploadLocation")
            put("longitude", lon)
            put("latitude", lat)
            put("deviceId", deviceId)
            put("status", status)
            put("timestamp", System.currentTimeMillis())
        }

        Log.d(TAG, ">>> 上传位置: lon=$lon, lat=$lat, deviceId=$deviceId, status=$status, retry=$retryCount")

        val request = Request.Builder()
            .url(uniCloudUrl)
            .post(json.toString().toRequestBody("application/json; charset=utf-8".toMediaType()))
            .addHeader("Content-Type", "application/json")
            .build()

        client.newCall(request).enqueue(object : Callback {
            override fun onFailure(call: Call, e: IOException) {
                Log.e(TAG, ">>> 数据上传失败 (retry=$retryCount): ${e.message}")
                if (retryCount < 2) {
                    Log.i(TAG, "3秒后重试上传...")
                    Handler(Looper.getMainLooper()).postDelayed({
                        uploadLocation(lon, lat, status, retryCount + 1)
                    }, 3000)
                } else {
                    Log.e(TAG, ">>> 上传失败，已达最大重试次数，放弃")
                }
            }

            override fun onResponse(call: Call, response: Response) {
                response.use {
                    val bodyStr = it.body?.string()
                    if (it.isSuccessful) {
                        Log.i(TAG, ">>> 数据上传成功: $bodyStr")
                    } else {
                        Log.e(TAG, ">>> 上传失败 HTTP ${it.code}: $bodyStr")
                    }
                }
            }
        })
    }
}
