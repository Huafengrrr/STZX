package com.example.myapplication1

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
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
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.Response
import org.json.JSONObject
import java.io.IOException
import java.util.Locale
import kotlin.math.cos
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * 语音导航管理器
 * 负责目的地查询、步行路线规划、逐步播报和到达节点检测
 */
class NavigationManager(
    private val context: Context,
    private val client: OkHttpClient,
    private val amapWebKey: String,
    private val getLastLocation: () -> Pair<Double, Double>,
    private val onSpeak: (text: String, isInterrupt: Boolean) -> Unit
) {
    companion object {
        private const val TAG = "Nav"
    }

    /**
     * 导航步骤数据类
     */
    data class NavigationStep(
        val instruction: String,
        val distance: Int,
        val roadName: String,
        val lat: Double,
        val lon: Double
    )

    var isNavigating = false
        private set
    private val navSteps = mutableListOf<NavigationStep>()
    private var currentStepIndex = 0

    /**
     * 导航入口：获取当前位置后规划步行路线
     * @param destination 用户说的目的地名称
     */
    fun startNavigation(destination: String) {
        onSpeak("好的，正在规划去$destination 的步行路线，请稍候", true)

        if (ContextCompat.checkSelfPermission(context, Manifest.permission.ACCESS_FINE_LOCATION)
            != PackageManager.PERMISSION_GRANTED) {
            onSpeak("缺少定位权限，无法导航，请在设置中开启", true)
            return
        }

        val locationManager = context.getSystemService(Context.LOCATION_SERVICE) as LocationManager
        val provider = if (locationManager.isProviderEnabled(LocationManager.GPS_PROVIDER)) {
            LocationManager.GPS_PROVIDER
        } else {
            LocationManager.NETWORK_PROVIDER
        }

        try {
            locationManager.requestLocationUpdates(provider, 0L, 0f, object : LocationListener {
                override fun onLocationChanged(loc: Location) {
                    locationManager.removeUpdates(this)
                    val (gcjLon, gcjLat) = StereoVision.wgs84ToGcj02(loc.longitude, loc.latitude)
                    fetchWalkingRoute(gcjLon, gcjLat, destination)
                }
                override fun onStatusChanged(provider: String?, status: Int, extras: Bundle?) {}
                override fun onProviderEnabled(provider: String) {}
                override fun onProviderDisabled(provider: String) {}
            }, Looper.getMainLooper())
        } catch (e: Exception) {
            Log.e(TAG, "导航定位失败", e)
            onSpeak("获取当前位置失败，导航无法启动", true)
        }
    }

    /**
     * 调用高德步行路线规划 Web API
     * 先将目的地名称转为坐标，再查询路线
     */
    private fun fetchWalkingRoute(originLon: Double, originLat: Double, destinationName: String) {
        val geoUrl = "https://restapi.amap.com/v3/geocode/geo?address=$destinationName&key=$amapWebKey"

        client.newCall(Request.Builder().url(geoUrl).get().build()).enqueue(object : Callback {
            override fun onFailure(call: Call, e: IOException) {
                onSpeak("网络连接失败，无法查询目的地", true)
            }

            override fun onResponse(call: Call, response: Response) {
                response.use {
                    if (!it.isSuccessful) {
                        onSpeak("查询目的地失败，请换一个地址试试", true)
                        return
                    }
                    try {
                        val geoJson = JSONObject(it.body?.string() ?: "")
                        if (geoJson.getString("status") != "1" || geoJson.getInt("count") == 0) {
                            onSpeak("未找到目的地，请换一个地址试试", true)
                            return
                        }

                        val destLocation = geoJson.getJSONArray("geocodes")
                            .getJSONObject(0).getString("location")
                        val destParts = destLocation.split(",")
                        val destLon = destParts[0].toDouble()
                        val destLat = destParts[1].toDouble()
                        val destName = geoJson.getJSONArray("geocodes")
                            .getJSONObject(0).optString("name", destinationName)

                        queryWalkingRoute(originLon, originLat, destLon, destLat, destName, destinationName)

                    } catch (e: Exception) {
                        Log.e(TAG, "解析目的地坐标失败", e)
                        onSpeak("解析目的地信息出错，请换一个地址试试", true)
                    }
                }
            }
        })
    }

    /**
     * 根据起终点坐标调用步行路线规划接口，解析并播报路线
     */
    private fun queryWalkingRoute(
        originLon: Double, originLat: Double,
        destLon: Double, destLat: Double,
        destName: String, destinationName: String
    ) {
        val routeUrl = "https://restapi.amap.com/v3/direction/walking" +
                "?origin=$originLon,$originLat&destination=$destLon,$destLat&key=$amapWebKey"

        client.newCall(Request.Builder().url(routeUrl).get().build()).enqueue(object : Callback {
            override fun onFailure(call: Call, e: IOException) {
                onSpeak("路线查询网络失败，请稍后重试", true)
            }

            override fun onResponse(call: Call, response: Response) {
                response.use {
                    if (!it.isSuccessful) {
                        onSpeak("路线规划失败，请稍后重试", true)
                        return
                    }
                    try {
                        val routeJson = JSONObject(it.body?.string() ?: "")
                        if (routeJson.getString("status") != "1") {
                            onSpeak("路线规划返回异常，请稍后重试", true)
                            return
                        }

                        val route = routeJson.getJSONObject("route")
                        val pathArray = route.getJSONArray("paths")
                        if (pathArray.length() == 0) {
                            onSpeak("未能规划出可行路线，请确认起终点是否正确", true)
                            return
                        }

                        val stepsArray = pathArray.getJSONObject(0).getJSONArray("steps")
                        navSteps.clear()
                        var totalDistance = 0

                        for (i in 0 until stepsArray.length()) {
                            val step = stepsArray.getJSONObject(i)
                            val instruction = step.getString("instruction")
                            val distance = step.getInt("distance")
                            val roadName = step.optString("road_name", "道路")
                            val stepLocation = step.optString("location", "")
                            val stepParts = if (stepLocation.contains(",")) {
                                stepLocation.split(",")
                            } else {
                                listOf(
                                    (originLon + (destLon - originLon) * (i + 1) / stepsArray.length()).toString(),
                                    (originLat + (destLat - originLat) * (i + 1) / stepsArray.length()).toString()
                                )
                            }
                            val stepLon = stepParts.getOrNull(0)?.toDoubleOrNull() ?: destLon
                            val stepLat = stepParts.getOrNull(1)?.toDoubleOrNull() ?: destLat

                            navSteps.add(NavigationStep(instruction, distance, roadName, stepLat, stepLon))
                            totalDistance += distance
                        }

                        isNavigating = true
                        currentStepIndex = 0

                        val walkTime = (totalDistance / 80).coerceAtLeast(1)
                        onSpeak("开始导航，终点：${destName}。全程约${totalDistance}米，预计步行约${walkTime}分钟。", true)

                        Handler(Looper.getMainLooper()).postDelayed({
                            announceCurrentNavStep()
                        }, 3000)

                    } catch (e: Exception) {
                        Log.e(TAG, "解析路线失败", e)
                        onSpeak("解析路线数据出错，请稍后重试", true)
                    }
                }
            }
        })
    }

    /**
     * 播报当前导航步骤
     */
    private fun announceCurrentNavStep() {
        if (!isNavigating || currentStepIndex >= navSteps.size) {
            if (isNavigating) {
                isNavigating = false
                navSteps.clear()
                onSpeak("已到达目的地！祝您一路平安！", true)
            }
            return
        }

        val step = navSteps[currentStepIndex]
        val distanceText = if (step.distance >= 1000) {
            String.format(Locale.US, "%.1f公里", step.distance / 1000.0)
        } else {
            "${step.distance}米"
        }

        val speakText = buildString {
            append("沿${step.roadName}直行")
            append(distanceText)
            append("，然后")
            append(step.instruction)
        }
        onSpeak(speakText, false)
    }

    /**
     * 在后台巡航时检查是否到达下一个导航节点（到达阈值 20 米）
     */
    fun checkNavProgress() {
        if (!isNavigating || currentStepIndex >= navSteps.size) return

        val (currentLon, currentLat) = getLastLocation()
        val step = navSteps[currentStepIndex]
        val distToNode = haversineDistance(currentLat, currentLon, step.lat, step.lon)

        if (distToNode <= 20.0) {
            currentStepIndex++
            if (currentStepIndex == navSteps.size - 1) {
                val lastStep = navSteps[currentStepIndex]
                onSpeak("前方${lastStep.distance}米，到达${lastStep.instruction}", false)
                currentStepIndex++
            } else if (currentStepIndex < navSteps.size) {
                announceCurrentNavStep()
            } else {
                announceCurrentNavStep()
            }
        }
    }

    /**
     * Haversine 半正矢公式计算两点间球面距离（米）
     */
    private fun haversineDistance(lat1: Double, lon1: Double, lat2: Double, lon2: Double): Double {
        val R = 6371000.0
        val dLat = Math.toRadians(lat2 - lat1)
        val dLon = Math.toRadians(lon2 - lon1)
        val a = sin(dLat / 2) * sin(dLat / 2) +
                cos(Math.toRadians(lat1)) * cos(Math.toRadians(lat2)) *
                sin(dLon / 2) * sin(dLon / 2)
        val c = 2 * Math.atan2(sqrt(a), sqrt(1 - a))
        return R * c
    }
}
