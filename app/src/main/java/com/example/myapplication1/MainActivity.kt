package com.example.myapplication1

import android.Manifest
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageFormat
import android.graphics.SurfaceTexture
import android.graphics.YuvImage
import android.location.Location
import android.location.LocationListener
import android.location.LocationManager
import android.os.Bundle
import android.speech.RecognizerIntent
import android.speech.tts.TextToSpeech
import android.util.Base64
import android.util.Log
import android.view.KeyEvent
import android.view.TextureView
import android.widget.ImageView
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import com.jiangdg.ausbc.CameraClient
import com.jiangdg.ausbc.callback.IPreviewDataCallBack
import com.jiangdg.ausbc.camera.CameraUvcStrategy
import com.jiangdg.ausbc.camera.bean.CameraRequest
import com.jiangdg.ausbc.widget.AspectRatioTextureView
import okhttp3.*
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONArray
import org.json.JSONObject
import org.opencv.android.OpenCVLoader
import org.opencv.calib3d.Calib3d
import org.opencv.calib3d.StereoSGBM
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.imgproc.Imgproc
import org.tensorflow.lite.Interpreter
import java.io.ByteArrayOutputStream
import java.io.FileInputStream
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.util.Locale
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import kotlin.math.abs
import kotlin.math.pow
import kotlin.math.sqrt
import androidx.core.splashscreen.SplashScreen.Companion.installSplashScreen

/**
 * 智能盲杖项目主活动类
 * 负责调度摄像头画面获取、YOLO目标检测、双目视觉测距、大语言模型交互、地理定位与语音播报。
 */
class MainActivity : AppCompatActivity() {

    /**
     * 目标检测数据类 (Data Class)
     * 用于存储 YOLO 模型推理后提取出的每一个物体的属性。
     * @param cx 边界框中心点 X 坐标
     * @param cy 边界框中心点 Y 坐标
     * @param w 边界框宽度
     * @param h 边界框高度
     * @param classId 类别索引 (对应 labels 数组)
     * @param score 置信度得分 (0.0 ~ 1.0)
     * @param distance 当前帧计算出的深度/距离 (米)
     * @param lastDistance 历史/上一帧的距离 (用于 EMA 滤波平滑)
     * @param disparity 🌟 新增：该物体的双目视差，用于右眼画面的精准画框偏移
     */
    data class Detection(var cx: Float, var cy: Float, var w: Float, var h: Float, val classId: Int, val score: Float, var distance: Float = -1f, var lastDistance: Float = -1f, var disparity: Float = 0f)

    // ===== 状态控制变量 =====
    private var lastVolumeClickTime = 0L // 记录上次按下音量键的时间，防抖
    private var latestBitmap: Bitmap? = null // 缓存最新的摄像头画面，供 LLM 分析使用

    // ===== API 密钥与地址 =====
    private val API_URL = "https://u755199-86f1-40c5b23f.westd.seetacloud.com:8443" // AutoDL Qwen-VL 模型接口地址
    private val MODEL_NAME = "qwen3-vl:8b" // 请求的大模型名称
    private val AMAP_WEB_KEY = "66eafa772acc05e7a8e587b4bd69074e" // 高德地图 Web 服务 API Key
    private val client = OkHttpClient() // OkHttp 网络请求客户端

    // ===== 🌟 导航与寻址专用的后台状态变量 =====
    private var locationTimer: java.util.Timer? = null // 后台静默定位定时器
    private var lastAnnouncedStreet = "" // 记录上次播报的街道名，防止一直重复同一条街
    private var lastIntersectionTime = 0L // 记录上次播报交叉路口的时间戳，防路口刷屏

    /**
     * 语音识别结果回调启动器
     * 当用户说话完毕，系统返回语音转文字的结果后触发。
     */
    private val voiceLauncher = registerForActivityResult(ActivityResultContracts.StartActivityForResult()) { result ->
        if (result.resultCode == RESULT_OK) {
            val matches = result.data?.getStringArrayListExtra(RecognizerIntent.EXTRA_RESULTS)
            if (!matches.isNullOrEmpty()) {
                val userText = matches[0] // 提取用户说的第一句话
                // 🌟 意图路由升级：扩大定位触发词库，支持更多自然语言问法
                val locationKeywords = listOf("在哪", "位置", "定位", "地方", "方位", "哪儿", "什么路", "什么街")
                if (locationKeywords.any { userText.contains(it) }) {
                    tellMeWhereIAm()
                } else {
                    // 🌟 按键触发，拥有最高打断特权，强行切断废话！
                    speakOut("收到，正在看图思考", isInterrupt = true)
                    askLlm(userText)
                }
            }
        }
    }

    // ===== UI与组件声明 =====
    private lateinit var fpsTextView: TextView
    private var frameCount = 0
    private var lastFpsTime = 0L
    private lateinit var fallDetector: FallDetector // 跌倒检测模块 (需外部实现)

    // 🌟 核心绑定：双目分屏相关的 UI 组件
    private lateinit var ivLeftEye: ImageView
    private lateinit var ivRightEye: ImageView
    private lateinit var overlayLeft: OverlayView
    private lateinit var overlayRight: OverlayView

    // ===== 测距、测速与平滑算法缓存 =====
    // 🌟 升级：双轨制语音控制时间戳，普通与紧急分离
    private var lastNormalSpeakTime = 0L // 控制普通语音播报频率，避免连续吵闹
    private var lastUrgentSpeakTime = 0L // 控制紧急语音防抖，确保每句警报能完整说完
    private var lastPrimaryDist = 0f // 主目标上一帧距离，用于测速
    private var lastPrimaryTime = 0L // 主目标上一帧时间，用于测速
    private var lastPrimaryCx = 0f
    private var lastPrimaryCy = 0f
    private var lastPrimaryClassId = -1
    private val speedBuffer = mutableListOf<Float>() // 测速缓存池，计算移动平均速度
    private var smoothedDistance = 0f // 主目标 EMA 滤波距离缓存
    private var lastSmoothedDetections = mutableListOf<Detection>() // 历史帧检测结果，用于 IoU 追踪和画框防抖

    // 🌟 核心修改 1：将英文标签替换为中文，否则 TTS 语音引擎播报时会很别扭
    private val labels = listOf(
        "人行道",  // 0: sidewalk
        "人",     // 1: person
        "柱子",   // 2: pole
        "汽车",   // 3: car
        "卡车",   // 4: truck
        "公交车", // 5: bus
        "自行车"  // 6: bicycle
    )

    // 🌟 核心修改 2：危险/大件物品白名单
    // 注意：去掉了"人行道(sidewalk)"，因为你肯定不想盲杖对盲人说“警告，距离人行道过近请避让”
    private val dangerousLabels = setOf(
        "人", "柱子", "汽车", "卡车", "公交车", "自行车"
    )

    // ===== OpenCV 双目相机标定参数 (内参、畸变、旋转平移矩阵) =====
    private val cameraMatrixL by lazy { Mat(3, 3, CvType.CV_64F).apply { put(0, 0, 328.26391515, 0.0, 327.21824491); put(1, 0, 0.0, 328.38426787, 235.88575854); put(2, 0, 0.0, 0.0, 1.0) } }
    private val distCoeffL by lazy { Mat(5, 1, CvType.CV_64F).apply { put(0, 0, 0.0706293, -0.0739329, -0.0660674, 0.0, 0.0) } }
    private val cameraMatrixR by lazy { Mat(3, 3, CvType.CV_64F).apply { put(0, 0, 329.07585782, 0.0, 312.23129337); put(1, 0, 0.0, 329.28633065, 237.06551395); put(2, 0, 0.0, 0.0, 1.0) } }
    private val distCoeffR by lazy { Mat(5, 1, CvType.CV_64F).apply { put(0, 0, 0.0767674, -0.0996155, 0.0, 0.0, 0.0) } }
    private val Rot by lazy { Mat(3, 3, CvType.CV_64F).apply { put(0, 0, 0.99996649, -0.00016997, 0.00818365); put(1, 0, 0.00017301, 0.99999991, -0.00037007); put(2, 0, -0.00818358, 0.00037147, 0.99996644) } }
    private val Trans by lazy { Mat(3, 1, CvType.CV_64F).apply { put(0, 0, -60.992415, -0.048313, -0.143001) } }

    // 立体校正映射矩阵缓存
    private val Rl by lazy { Mat() }; private val Rr by lazy { Mat() }
    private val Pl by lazy { Mat() }; private val Pr by lazy { Mat() }
    private val Q by lazy { Mat() }
    private val mapLx by lazy { Mat() }; private val mapLy by lazy { Mat() }
    private val mapRx by lazy { Mat() }; private val mapRy by lazy { Mat() }

    // 🌟 SGBM (半全局块匹配) 算法配置：用于生成视差图
    private val sgbm by lazy {
        val blockWindowSize = 11 // 块匹配窗口大小，必须是奇数
        StereoSGBM.create(0, 64, blockWindowSize).apply {
            setP1(8 * 1 * blockWindowSize * blockWindowSize) // 惩罚系数1，控制平滑度
            setP2(32 * 1 * blockWindowSize * blockWindowSize) // 惩罚系数2，控制平滑度(P2>P1)
            setPreFilterCap(63) // 预处理滤波器截断值
            setMode(StereoSGBM.MODE_SGBM_3WAY) // 使用 3 路径匹配
            setUniquenessRatio(10) // 唯一性比率，过滤掉非唯一匹配(降噪)
            setSpeckleWindowSize(100) // 斑点过滤窗口，用于填补视差图中的小黑洞
            setSpeckleRange(32) // 斑点视差变化阈值
            setDisp12MaxDiff(1) // 左右视差一致性检测最大容许误差
        }
    }

    private lateinit var mCameraView: AspectRatioTextureView
    private var mCameraClient: CameraClient? = null
    private lateinit var cameraExecutor: ExecutorService
    private lateinit var tts: TextToSpeech
    private lateinit var tflite: Interpreter

    // 声明应用所需的所有权限 (相机、录音、精准定位、粗略定位)
    private val requiredPermissions = arrayOf(
        Manifest.permission.CAMERA, Manifest.permission.RECORD_AUDIO,
        Manifest.permission.ACCESS_FINE_LOCATION, Manifest.permission.ACCESS_COARSE_LOCATION
    )

    // 动态权限申请回调
    private val requestPermissionLauncher = registerForActivityResult(ActivityResultContracts.RequestMultiplePermissions()) {
        if (it.entries.all { p -> p.value }) {
            startUSBCamera()
            startLocationCruise() // 权限授予后，启动后台巡航定位
        } else {
            Toast.makeText(this, "需要全部权限才能正常运行", Toast.LENGTH_SHORT).show()
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        // 🌟 核心修改：在 super.onCreate 之前调用，开启官方开机界面
        val splashScreen = installSplashScreen()

        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        // 初始化 OpenCV
        if (!OpenCVLoader.initDebug()) Log.e("OpenCV", "OpenCV 初始化失败！") else initStereoRectify()

        // 🌟 核心绑定：左眼/右眼画面 以及对应的 画框图层
        ivLeftEye = findViewById(R.id.iv_left_eye)
        ivRightEye = findViewById(R.id.iv_right_eye)
        overlayLeft = findViewById(R.id.overlay_left)
        overlayRight = findViewById(R.id.overlay_right)
        fpsTextView = findViewById(R.id.fps_text_view)

        lastFpsTime = System.currentTimeMillis()

        mCameraView = findViewById(R.id.camera_view)
        mCameraView.alpha = 0f // 隐藏原始相机预览层
        overlayLeft.bringToFront() // 将画框层置于顶层
        overlayRight.bringToFront()

        cameraExecutor = Executors.newSingleThreadExecutor()

        // 初始化 TTS 语音引擎
        tts = TextToSpeech(this) {
            if (it == TextToSpeech.SUCCESS) {
                tts.setLanguage(Locale.CHINESE)
                speakOut("声途智行启动成功", isInterrupt = false)
            }
        }

        // 初始化跌倒检测器
        fallDetector = FallDetector(this) {
            runOnUiThread { Toast.makeText(this, "检测到严重跌倒！", Toast.LENGTH_LONG).show() }
            // 🌟 跌倒属于最高优先级，直接强制掐断当前所有语音并强行插播
            speakOut("警告！检测到摔倒，正在启动紧急求助程序！", isInterrupt = true)
        }
        fallDetector.start()

        // 加载 YOLOv8 TFLite 模型，开启 4 线程加速
        try {
            tflite = Interpreter(loadModelFile("yolo.tflite"), Interpreter.Options().apply { numThreads = 4 })
        } catch (e: Exception) { Log.e("TFLite", "模型加载失败", e) }

        // 检查并请求权限
        if (requiredPermissions.all { ContextCompat.checkSelfPermission(this, it) == PackageManager.PERMISSION_GRANTED }) {
            startUSBCamera()
            startLocationCruise() // 🌟 权限都有了，启动后台位置巡航
        } else {
            requestPermissionLauncher.launch(requiredPermissions)
        }
    }

    // ================== 🌟 智能导航与寻址核心逻辑 ==================

    /**
     * 启动后台位置巡航定时器
     * 每 15 秒静默获取一次位置，检查是否进入新街道或到达交叉路口
     */
    private fun startLocationCruise() {
        locationTimer = java.util.Timer()
        locationTimer?.scheduleAtFixedRate(object : java.util.TimerTask() {
            override fun run() {
                // 静默获取经纬度，不触发语音 "正在获取位置..."
                silentlyFetchLocation()
            }
        }, 5000, 15000) // 延迟5秒启动，之后每 15 秒执行一次
    }

    /**
     * 后台静默抓取一次位置
     */
    private fun silentlyFetchLocation() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.ACCESS_FINE_LOCATION) != PackageManager.PERMISSION_GRANTED) return
        val locationManager = getSystemService(Context.LOCATION_SERVICE) as LocationManager

        // 优先使用最高精度的 GPS 缓存，其次才是网络基站
        val location = locationManager.getLastKnownLocation(LocationManager.GPS_PROVIDER)
            ?: locationManager.getLastKnownLocation(LocationManager.NETWORK_PROVIDER)

        if (location != null) {
            // 🌟 核心修复：后台巡航也要进行火星坐标系转换，消除 500 米物理偏移！
            val (gcjLon, gcjLat) = wgs84ToGcj02(location.longitude, location.latitude)
            fetchAddressFromAmap(gcjLon, gcjLat, isSilent = true)
        }
    }

    /**
     * 用户主动询问位置
     */
    private fun tellMeWhereIAm() {
        // 🌟 按键触发的定位询问，同样具备最高打断特权
        speakOut("正在获取精准定位，请稍候", isInterrupt = true)
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.ACCESS_FINE_LOCATION) != PackageManager.PERMISSION_GRANTED) {
            speakOut("缺少定位权限，请在设置中开启", isInterrupt = true)
            return
        }

        val locationManager = getSystemService(Context.LOCATION_SERVICE) as LocationManager

        // 🌟 强行优先使用 GPS_PROVIDER 获取最高精度的卫星定位！如果室内无 GPS 信号再降级使用网络
        val provider = if (locationManager.isProviderEnabled(LocationManager.GPS_PROVIDER)) {
            LocationManager.GPS_PROVIDER
        } else {
            LocationManager.NETWORK_PROVIDER
        }

        // 强制向系统发起一次最新的网络定位请求，确保地点绝对准确！
        locationManager.requestLocationUpdates(provider, 0L, 0f, object : LocationListener {
            override fun onLocationChanged(loc: Location) {
                locationManager.removeUpdates(this) // 获取到最新位置后立即移除监听，省电

                // 🌟 终极修复：把手机原生定位 (WGS84 国际标准) 转换为高德地图专属坐标 (GCJ-02 火星坐标)
                // 彻底解决地点完全对不上、漂移好几条街的深坑！
                val (gcjLon, gcjLat) = wgs84ToGcj02(loc.longitude, loc.latitude)

                fetchAddressFromAmap(gcjLon, gcjLat, isSilent = false)
            }
            override fun onStatusChanged(provider: String?, status: Int, extras: Bundle?) {}
            override fun onProviderEnabled(provider: String) {}
            override fun onProviderDisabled(provider: String) {}
        })
    }

    /**
     * 访问高德 API 并进行智能状态分发播报
     * @param isSilent: true 代表是后台自动巡航查询，只有在关键节点才发声；false 代表用户主动提问，必须发声。
     */
    private fun fetchAddressFromAmap(lon: Double, lat: Double, isSilent: Boolean = false) {
        // 🌟 精度保卫战：强行将转换后的经纬度格式化为 6 位小数（亚米级精度），绝不让高德收到低精度截断坐标！
        val formattedLon = String.format(Locale.US, "%.6f", lon)
        val formattedLat = String.format(Locale.US, "%.6f", lat)

        val url = "https://restapi.amap.com/v3/geocode/regeo?location=$formattedLon,$formattedLat&extensions=all&key=$AMAP_WEB_KEY"

        client.newCall(Request.Builder().url(url).get().build()).enqueue(object : Callback {
            override fun onFailure(call: Call, e: IOException) {
                if (!isSilent) speakOut("网络连接失败，无法查询位置", isInterrupt = true)
            }
            override fun onResponse(call: Call, response: Response) {
                response.use {
                    if (!it.isSuccessful) {
                        if (!isSilent) speakOut("定位服务器返回错误", isInterrupt = true)
                        return
                    }
                    try {
                        val json = JSONObject(it.body?.string() ?: "")
                        if (json.getString("status") == "1") {
                            val regeocode = json.getJSONObject("regeocode")
                            val addressComponent = regeocode.getJSONObject("addressComponent")

                            // 1. 获取当前所在的街道名字
                            var currentStreet = ""
                            val streetNumber = addressComponent.optJSONObject("streetNumber")
                            if (streetNumber != null && streetNumber.has("street") && !streetNumber.isNull("street")) {
                                val streetStr = streetNumber.getString("street")
                                if (streetStr.isNotEmpty() && streetStr != "[]") {
                                    currentStreet = streetStr
                                }
                            }

                            // 2. 获取完整的长地址
                            val fullAddress = regeocode.getString("formatted_address")

                            // 3. 交叉路口预警提取
                            var intersectionAlert = ""
                            try {
                                if (regeocode.has("roadinters")) {
                                    val intersElement = regeocode.get("roadinters")
                                    var intersArray: JSONArray? = null

                                    if (intersElement is JSONArray) {
                                        intersArray = intersElement
                                    } else if (intersElement is String && intersElement.startsWith("[")) {
                                        intersArray = JSONArray(intersElement)
                                    }

                                    if (intersArray != null && intersArray.length() > 0) {
                                        val nearestInter = intersArray.getJSONObject(0)
                                        val distStr = nearestInter.optString("distance", "999")
                                        val distToInter = distStr.toDoubleOrNull() ?: 999.0

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

                            // 4. 周边道路方位提取 (方案一：绝对方向播报)
                            var roadsAlert = ""
                            try {
                                if (regeocode.has("roads")) {
                                    val roadsElement = regeocode.get("roads")
                                    var roadsArray: JSONArray? = null

                                    if (roadsElement is JSONArray) {
                                        roadsArray = roadsElement
                                    } else if (roadsElement is String && roadsElement.startsWith("[")) {
                                        roadsArray = JSONArray(roadsElement)
                                    }

                                    if (roadsArray != null && roadsArray.length() > 0) {
                                        val roadInfoList = mutableListOf<String>()
                                        // 只提取最近的两条道路信息，避免太啰嗦
                                        val maxRoads = minOf(2, roadsArray.length())
                                        for (i in 0 until maxRoads) {
                                            val road = roadsArray.getJSONObject(i)
                                            val roadName = road.optString("name", "")
                                            val direction = road.optString("direction", "")

                                            // 兼容高德返回的异常格式
                                            if (roadName.isNotEmpty() && direction.isNotEmpty() && direction != "[]") {
                                                // 让方向更加口语化，把"东"变成"东侧"
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

                            // 5. 智能播报路由
                            if (!isSilent) {
                                // 拼接路口和周边道路信息
                                val speakText = buildString {
                                    append("您当前位于：$fullAddress。")
                                    if (intersectionAlert.isNotEmpty()) append(intersectionAlert)
                                    if (roadsAlert.isNotEmpty()) append(roadsAlert) // 只在用户主动提问时才播报周边道路
                                }
                                // 🌟 用户主动提问的反馈，特权打断
                                speakOut(speakText.toString(), isInterrupt = true)
                            } else {
                                val currentTime = System.currentTimeMillis()

                                // 巡航模式：只报街道变更或路口，坚决不报周边道路，防止话痨
                                if (intersectionAlert.isNotEmpty() && (currentTime - lastIntersectionTime > 60000L)) {
                                    // 巡航信息不使用打断，老老实实排队
                                    speakOut("注意，$intersectionAlert", isInterrupt = false)
                                    lastIntersectionTime = currentTime
                                }
                                else if (currentStreet.isNotEmpty() && currentStreet != lastAnnouncedStreet) {
                                    if (lastAnnouncedStreet.isNotEmpty()) {
                                        speakOut("您已进入 $currentStreet", isInterrupt = false)
                                    }
                                    lastAnnouncedStreet = currentStreet
                                }
                            }
                        } else {
                            if (!isSilent) speakOut("未能获取详细街道信息", isInterrupt = true)
                        }
                    } catch (e: Exception) {
                        Log.e("AMAP", "高德解析错误", e)
                        if (!isSilent) speakOut("解析位置数据出错了", isInterrupt = true)
                    }
                }
            }
        })
    }

    // =============================================================
    // 🌟 【重磅武器】：WGS84 国际标准坐标 转 GCJ-02 火星坐标算法
    // 彻底消灭国内安卓手机定位和高德地图之间 500 米偏移的罪魁祸首！

    private fun transformLat(x: Double, y: Double): Double {
        var ret = -100.0 + 2.0 * x + 3.0 * y + 0.2 * y * y + 0.1 * x * y + 0.2 * Math.sqrt(Math.abs(x))
        ret += (20.0 * Math.sin(6.0 * x * Math.PI) + 20.0 * Math.sin(2.0 * x * Math.PI)) * 2.0 / 3.0
        ret += (20.0 * Math.sin(y * Math.PI) + 40.0 * Math.sin(y / 3.0 * Math.PI)) * 2.0 / 3.0
        ret += (160.0 * Math.sin(y / 12.0 * Math.PI) + 320.0 * Math.sin(y * Math.PI / 30.0)) * 2.0 / 3.0
        return ret
    }

    private fun transformLon(x: Double, y: Double): Double {
        var ret = 300.0 + x + 2.0 * y + 0.1 * x * x + 0.1 * x * y + 0.1 * Math.sqrt(Math.abs(x))
        ret += (20.0 * Math.sin(6.0 * x * Math.PI) + 20.0 * Math.sin(2.0 * x * Math.PI)) * 2.0 / 3.0
        ret += (20.0 * Math.sin(x * Math.PI) + 40.0 * Math.sin(x / 3.0 * Math.PI)) * 2.0 / 3.0
        ret += (150.0 * Math.sin(x / 12.0 * Math.PI) + 300.0 * Math.sin(x / 30.0 * Math.PI)) * 2.0 / 3.0
        return ret
    }

    /**
     * 将国际标准 GPS 坐标 (WGS84) 转换为高德地图特有的火星坐标系 (GCJ02)
     */
    private fun wgs84ToGcj02(lon: Double, lat: Double): Pair<Double, Double> {
        // 如果不在中国境内，无需转换直接返回
        if (lon < 72.004 || lon > 137.8347 || lat < 0.8293 || lat > 55.8271) return Pair(lon, lat)

        val a = 6378245.0
        val ee = 0.00669342162296594323
        var dLat = transformLat(lon - 105.0, lat - 35.0)
        var dLon = transformLon(lon - 105.0, lat - 35.0)
        val radLat = lat / 180.0 * Math.PI
        var magic = Math.sin(radLat)
        magic = 1 - ee * magic * magic
        val sqrtMagic = Math.sqrt(magic)
        dLat = (dLat * 180.0) / ((a * (1 - ee)) / (magic * sqrtMagic) * Math.PI)
        dLon = (dLon * 180.0) / (a / sqrtMagic * Math.cos(radLat) * Math.PI)

        return Pair(lon + dLon, lat + dLat)
    }

    // =============================================================

    /**
     * 计算两个边界框的交并比 (Intersection over Union)，用于 NMS 和物体追踪
     */
    private fun calculateIoU(box1: Detection, box2: Detection): Float {
        val x1 = maxOf(box1.cx - box1.w / 2f, box2.cx - box2.w / 2f)
        val y1 = maxOf(box1.cy - box1.h / 2f, box2.cy - box2.h / 2f)
        val x2 = minOf(box1.cx + box1.w / 2f, box2.cx + box2.w / 2f)
        val y2 = minOf(box1.cy + box1.h / 2f, box2.cy + box2.h / 2f)
        val intersectionArea = maxOf(0f, x2 - x1) * maxOf(0f, y2 - y1)
        val box1Area = box1.w * box1.h
        val box2Area = box2.w * box2.h
        return intersectionArea / (box1Area + box2Area - intersectionArea)
    }

    /**
     * 初始化双目摄像头的立体校正映射矩阵。
     */
    private fun initStereoRectify() {
        val imageSize = org.opencv.core.Size(640.0, 480.0)
        Calib3d.stereoRectify(cameraMatrixL, distCoeffL, cameraMatrixR, distCoeffR, imageSize, Rot, Trans, Rl, Rr, Pl, Pr, Q, Calib3d.CALIB_ZERO_DISPARITY, 0.0, imageSize, null, null)
        Calib3d.initUndistortRectifyMap(cameraMatrixL, distCoeffL, Rl, Pl, imageSize, CvType.CV_32FC1, mapLx, mapLy)
        Calib3d.initUndistortRectifyMap(cameraMatrixR, distCoeffR, Rr, Pr, imageSize, CvType.CV_32FC1, mapRx, mapRy)
    }

    /**
     * 开启 USB 摄像头，绑定预览回调函数
     */
    private fun startUSBCamera() {
        mCameraClient = CameraClient.newBuilder(this).setCameraStrategy(CameraUvcStrategy(this)).setCameraRequest(CameraRequest.Builder().setPreviewWidth(1280).setPreviewHeight(480).create()).build()

        mCameraView.surfaceTextureListener = object : TextureView.SurfaceTextureListener {
            override fun onSurfaceTextureAvailable(s: SurfaceTexture, w: Int, h: Int) { mCameraClient?.openCamera(mCameraView) }
            override fun onSurfaceTextureSizeChanged(s: SurfaceTexture, w: Int, h: Int) {}
            override fun onSurfaceTextureDestroyed(s: SurfaceTexture): Boolean { mCameraClient?.closeCamera(); return true }
            override fun onSurfaceTextureUpdated(s: SurfaceTexture) {}
        }

        mCameraClient?.addPreviewDataCallBack(object : IPreviewDataCallBack {
            private var isDrawing = false
            private var isInferencing = false // 防止推理线程堆积的锁

            override fun onPreviewData(data: ByteArray?, format: IPreviewDataCallBack.DataFormat) {
                if (data == null || isDrawing) return
                isDrawing = true

                cameraExecutor.execute {
                    // ================= 🌟 修复 FPS 显示 Bug =================
                    frameCount++
                    val currentTime = System.currentTimeMillis()
                    if (currentTime - lastFpsTime >= 1000L) {
                        val currentFps = frameCount // 提前保存，防止跨线程清零
                        runOnUiThread { fpsTextView.text = "FPS: $currentFps" }
                        frameCount = 0
                        lastFpsTime = currentTime
                    }

                    try {
                        // 🌟 【防黑屏自适应算法】动态推算相机的真实分辨率，防止摄像头意外降级导致解析越界崩溃！
                        val dataSize = data.size
                        val actualWidth: Int
                        val actualHeight: Int

                        if (dataSize == 1280 * 480 * 3 / 2) {
                            actualWidth = 1280
                            actualHeight = 480
                        } else if (dataSize == 640 * 480 * 3 / 2) {
                            actualWidth = 640
                            actualHeight = 480
                        } else {
                            // 未知分辨率兜底跳过，防止解析越界崩溃导致黑屏
                            isDrawing = false
                            return@execute
                        }

                        val yuvImage = YuvImage(data, ImageFormat.NV21, actualWidth, actualHeight, null)
                        val out = ByteArrayOutputStream()
                        yuvImage.compressToJpeg(android.graphics.Rect(0, 0, actualWidth, actualHeight), 80, out)

                        val originalBitmap = BitmapFactory.decodeByteArray(out.toByteArray(), 0, out.size())
                        if (originalBitmap == null) {
                            isDrawing = false
                            return@execute
                        }

                        val leftPreviewBitmap: Bitmap
                        val rightPreviewBitmap: Bitmap

                        if (actualWidth >= 1280) {
                            leftPreviewBitmap = Bitmap.createBitmap(originalBitmap, 0, 0, 640, 480)
                            rightPreviewBitmap = Bitmap.createBitmap(originalBitmap, 640, 0, 640, 480)
                        } else {
                            // 如果摄像头意外降级到单目 640x480，左右眼画面使用同一张图兜底，绝对不黑屏！
                            leftPreviewBitmap = originalBitmap
                            rightPreviewBitmap = originalBitmap
                        }

                        latestBitmap = leftPreviewBitmap

                        // 🌟 解除卡顿封印：让上下两屏原画面全速 30 FPS 刷新上屏！
                        runOnUiThread {
                            ivLeftEye.setImageBitmap(leftPreviewBitmap)
                            ivRightEye.setImageBitmap(rightPreviewBitmap)
                        }

                        // 让 AI 推理部分在后台单独加锁慢慢算（3~5 FPS）
                        if (!isInferencing) {
                            isInferencing = true
                            Thread {
                                val fullMat = Mat(); val leftMat = Mat(); val rightMat = Mat()
                                val grayL = Mat(); val grayR = Mat(); val rectL = Mat(); val rectR = Mat()
                                val disparity16S = Mat(); val disparity32F = Mat()

                                try {
                                    // ================= 步骤 1: YOLO 推理与预处理 =================
                                    val paddedBitmap = Bitmap.createBitmap(640, 640, Bitmap.Config.ARGB_8888)
                                    val canvas = android.graphics.Canvas(paddedBitmap)
                                    canvas.drawColor(android.graphics.Color.rgb(114, 114, 114)) // 画布涂成标准灰色
                                    val paddingY = (640f - 480f) / 2f
                                    canvas.drawBitmap(leftPreviewBitmap, 0f, paddingY, null)

                                    val inputBuffer = ByteBuffer.allocateDirect(1 * 640 * 640 * 3 * 4)
                                    inputBuffer.order(ByteOrder.nativeOrder())
                                    val intValues = IntArray(640 * 640)
                                    paddedBitmap.getPixels(intValues, 0, 640, 0, 0, 640, 640)
                                    for (pixelValue in intValues) {
                                        inputBuffer.putFloat(((pixelValue shr 16) and 0xFF) / 255.0f)
                                        inputBuffer.putFloat(((pixelValue shr 8) and 0xFF) / 255.0f)
                                        inputBuffer.putFloat((pixelValue and 0xFF) / 255.0f)
                                    }

                                    val outputArray = Array(1) { Array(11) { FloatArray(8400) } }
                                    tflite.run(inputBuffer, outputArray)

                                    // ================= 步骤 2: YOLO 后处理 (解析输出并筛选) =================
                                    val rawDetections = mutableListOf<Detection>()
                                    for (i in 0 until 8400) {
                                        var maxClassScore = 0f
                                        var classId = -1
                                        for (c in 0 until 7) {
                                            if (outputArray[0][c + 4][i] > maxClassScore) {
                                                maxClassScore = outputArray[0][c + 4][i]
                                                classId = c
                                            }
                                        }

                                        val threshold = if (classId == 1) 0.20f else 0.30f

                                        if (maxClassScore > threshold) {
                                            rawDetections.add(Detection(
                                                outputArray[0][0][i], outputArray[0][1][i],
                                                outputArray[0][2][i], outputArray[0][3][i],
                                                classId, maxClassScore
                                            ))
                                        }
                                    }

                                    rawDetections.sortByDescending { it.score }
                                    val nmsDetections = mutableListOf<Detection>()
                                    for (det in rawDetections) {
                                        var suppress = false
                                        // 🌟 优化修复 1：彻底解决一个物体上重叠多个框！
                                        for (keep in nmsDetections) {
                                            if (calculateIoU(det, keep) > 0.45f) {
                                                suppress = true; break
                                            }
                                        }
                                        if (!suppress) nmsDetections.add(det)
                                    }

                                    // 🌟 优化修复 2：减少屏幕视觉杂乱，最多只保留得分最高的 4 个物体进行渲染
                                    val finalDetections = nmsDetections.take(4)
                                    val boxesToDrawLeft = mutableListOf<BoundingBox>()  // 🌟 左眼专用画框列表
                                    val boxesToDrawRight = mutableListOf<BoundingBox>() // 🌟 右眼专用画框列表

                                    if (finalDetections.isNotEmpty()) {
                                        val validDetections = mutableListOf<Detection>()
                                        for (det in finalDetections) {
                                            if (det.cx <= 1.5f && det.cy <= 1.5f) {
                                                det.cx *= 640f; det.cy *= 640f; det.w *= 640f; det.h *= 640f
                                            }
                                            det.cy -= 80f
                                            if (det.cy >= 0f && det.cy <= 480f) {
                                                validDetections.add(det)
                                            }
                                        }

                                        // ================= 步骤 3: 目标追踪与边界框平滑 (EMA 滤波) =================
                                        val boxAlpha = 0.3f
                                        for (det in validDetections) {
                                            var bestMatch: Detection? = null
                                            var maxIou = 0f
                                            for (oldDet in lastSmoothedDetections) {
                                                if (oldDet.classId == det.classId) {
                                                    val iou = calculateIoU(det, oldDet)
                                                    if (iou > 0.4f && iou > maxIou) {
                                                        maxIou = iou
                                                        bestMatch = oldDet
                                                    }
                                                }
                                            }

                                            if (bestMatch != null) {
                                                det.cx = boxAlpha * det.cx + (1f - boxAlpha) * bestMatch.cx
                                                det.cy = boxAlpha * det.cy + (1f - boxAlpha) * bestMatch.cy
                                                det.w = boxAlpha * det.w + (1f - boxAlpha) * bestMatch.w
                                                det.h = boxAlpha * det.h + (1f - boxAlpha) * bestMatch.h
                                                det.lastDistance = bestMatch.distance
                                            }
                                        }

                                        // ================= 步骤 4: SGBM 双目视差图生成 =================
                                        org.opencv.android.Utils.bitmapToMat(originalBitmap, fullMat)
                                        if (actualWidth >= 1280) {
                                            leftMat.apply { fullMat.submat(org.opencv.core.Rect(0, 0, 640, 480)).copyTo(this) }
                                            rightMat.apply { fullMat.submat(org.opencv.core.Rect(640, 0, 640, 480)).copyTo(this) }
                                        } else {
                                            leftMat.apply { fullMat.copyTo(this) }
                                            rightMat.apply { fullMat.copyTo(this) }
                                        }

                                        Imgproc.cvtColor(leftMat, grayL, Imgproc.COLOR_RGB2GRAY)
                                        Imgproc.cvtColor(rightMat, grayR, Imgproc.COLOR_RGB2GRAY)
                                        Imgproc.remap(grayL, rectL, mapLx, mapLy, Imgproc.INTER_LINEAR)
                                        Imgproc.remap(grayR, rectR, mapRx, mapRy, Imgproc.INTER_LINEAR)

                                        sgbm.compute(rectL, rectR, disparity16S)
                                        disparity16S.convertTo(disparity32F, CvType.CV_32F, 1.0 / 16.0)

                                        // ================= 步骤 5: 全量深度采样与距离计算 =================
                                        for (det in validDetections) {
                                            val safeX = det.cx.toInt().coerceIn(0, 639)
                                            val safeY = det.cy.toInt().coerceIn(0, 479)

                                            // 🌟【核心修复】左侧盲区智能宽幅采样算法：
                                            val searchW = (det.w * 0.40f).toInt().coerceAtLeast(10)
                                            val searchH = (det.h * 0.15f).toInt().coerceAtLeast(10)

                                            val startX = (safeX - searchW).coerceIn(0, 639)
                                            val endX = (safeX + searchW).coerceIn(0, 639)
                                            val startY = (safeY - searchH).coerceIn(0, 479)
                                            val endY = (safeY + searchH).coerceIn(0, 479)

                                            val dispList = mutableListOf<Double>()

                                            val stepX = maxOf(2, (endX - startX) / 10)
                                            val stepY = maxOf(2, (endY - startY) / 10)

                                            for (py in startY..endY step stepY) {
                                                for (px in startX..endX step stepX) {
                                                    val disp = disparity32F.get(py, px)[0]
                                                    if (disp > 1.0) {
                                                        dispList.add(disp)
                                                    }
                                                }
                                            }

                                            val d = if (dispList.size >= 3) {
                                                dispList.sort()
                                                val startIdx = dispList.size / 4
                                                val endIdx = dispList.size * 3 / 4
                                                if (endIdx > startIdx) {
                                                    dispList.subList(startIdx, endIdx).average()
                                                } else {
                                                    dispList[dispList.size / 2]
                                                }
                                            } else { 0.0 }

                                            if (d > 1.0) {
                                                det.disparity = d.toFloat()

                                                val rawZFloat = (((328.26 * 60.99) / d) / 1000.0).toFloat()
                                                if (rawZFloat > 0.2f && rawZFloat < 12.0f) {
                                                    if (det.lastDistance > 0.2f) {
                                                        val jump = abs(rawZFloat - det.lastDistance)
                                                        val alpha = if (jump > 1.5f) 0.1f else 0.3f
                                                        det.distance = (1f - alpha) * det.lastDistance + alpha * rawZFloat
                                                    } else {
                                                        det.distance = rawZFloat
                                                    }
                                                }
                                            }
                                        }

                                        // ================= 步骤 6: 距离过滤、排序、渲染与播报 =================
                                        val viewW = overlayLeft.width.toFloat()
                                        val viewH = overlayLeft.height.toFloat()

                                        val closeTargets = validDetections
                                            .filter { it.distance > 0.2f && it.distance <= 12.0f }
                                            .sortedBy { it.distance }

                                        for ((index, det) in closeTargets.withIndex()) {
                                            val finalLabel = if (det.classId < labels.size) labels[det.classId] else "未知"
                                            val finalZ = det.distance
                                            val distanceStr = String.format("%.1f米", finalZ)
                                            var speedStr = ""
                                            var isTooClose = false

                                            if (finalZ <= 1.0f && dangerousLabels.contains(finalLabel)) {
                                                isTooClose = true
                                            }

                                            // 语音永远只判定离你最近的那一个（index == 0）
                                            if (index == 0) {
                                                val currentMeasureTime = System.currentTimeMillis()
                                                if (lastPrimaryTime > 0L && lastPrimaryDist > 0f) {
                                                    val dt = (currentMeasureTime - lastPrimaryTime) / 1000.0f
                                                    if (dt > 0.05f && dt < 1.0f) {
                                                        val rawSpeed = (lastPrimaryDist - finalZ) / dt
                                                        if (abs(rawSpeed) < 10.0f) {
                                                            speedBuffer.add(rawSpeed)
                                                            if (speedBuffer.size > 5) speedBuffer.removeAt(0)
                                                            val avgSpeedMs = speedBuffer.average().toFloat()
                                                            if (avgSpeedMs > 0.5f) {
                                                                speedStr = String.format("，正以%.1fm/s靠近", avgSpeedMs)
                                                            }
                                                        }
                                                    }
                                                }
                                                lastPrimaryDist = finalZ
                                                lastPrimaryTime = currentMeasureTime
                                                lastPrimaryCx = det.cx
                                                lastPrimaryCy = det.cy
                                                lastPrimaryClassId = det.classId

                                                val currentTimeSpeak = System.currentTimeMillis()

                                                // 🌟 核心功能升级：【所有播报都必须等上一句话播完再播下一句】！
                                                if (isTooClose) {
                                                    // 🚨 紧急避让警报
                                                    // 必须等 tts 不在发音 (!tts.isSpeaking)，并且过了防抖冷却时间，才会播报下一句警告，绝不结巴！
                                                    if (!tts.isSpeaking && (currentTimeSpeak - lastUrgentSpeakTime > 2000L)) {
                                                        val speakText = "警告！距离$finalLabel 过近，请避让！"
                                                        speakOut(speakText, isInterrupt = false) // 老老实实排队
                                                        lastUrgentSpeakTime = currentTimeSpeak
                                                        lastNormalSpeakTime = currentTimeSpeak // 刷新普通时间
                                                    }
                                                } else {
                                                    // 🚶 常规路况播报
                                                    // 同样必须等上一句话说完，绝不插嘴
                                                    if (!tts.isSpeaking && (currentTimeSpeak - lastNormalSpeakTime > 2500L)) {
                                                        val speakText = if (speedStr.isNotEmpty()) {
                                                            "注意！最近的$finalLabel 在$distanceStr$speedStr"
                                                        } else {
                                                            "前方 $distanceStr 有$finalLabel"
                                                        }
                                                        speakOut(speakText, isInterrupt = false)
                                                        lastNormalSpeakTime = currentTimeSpeak
                                                    }
                                                }
                                            }

                                            val displayLabel = if (distanceStr.isNotEmpty()) "$finalLabel $distanceStr$speedStr" else finalLabel

                                            // 🌟 核心物理平移逻辑：分别为左眼和右眼计算不同的精准画框坐标！
                                            if (viewW > 0f && viewH > 0f) {
                                                val renderedH = viewW * (480f / 640f)
                                                val yOffset = (viewH - renderedH) / 2f
                                                val scaleFactor = viewW / 640f

                                                // 1. 左眼坐标 (标准 YOLO 坐标)
                                                val finalCxLeft = det.cx * scaleFactor
                                                val finalCy = det.cy * scaleFactor + yOffset
                                                val finalW = det.w * scaleFactor
                                                val finalH = det.h * scaleFactor
                                                boxesToDrawLeft.add(BoundingBox(finalCxLeft - finalW/2f, finalCy - finalH/2f, finalCxLeft + finalW/2f, finalCy + finalH/2f, displayLabel, det.score))

                                                // 2. 右眼坐标 (减去视差 Disparity 实现完美的物理偏移)
                                                val finalCxRight = (det.cx - det.disparity) * scaleFactor
                                                boxesToDrawRight.add(BoundingBox(finalCxRight - finalW/2f, finalCy - finalH/2f, finalCxRight + finalW/2f, finalCy + finalH/2f, displayLabel, det.score))
                                            }
                                        }

                                        // ================= 步骤 7: 保存全量历史记录 =================
                                        lastSmoothedDetections = validDetections.map { it.copy() }.toMutableList()

                                        // 将处理好的左右眼红框，同步更新到主界面的两个 Overlay 上
                                        runOnUiThread {
                                            overlayLeft.setResults(boxesToDrawLeft)
                                            overlayRight.setResults(boxesToDrawRight)
                                        }

                                    } else {
                                        // 无目标时同时清空左右两屏的画框
                                        runOnUiThread {
                                            overlayLeft.setResults(emptyList())
                                            overlayRight.setResults(emptyList())
                                        }
                                    }

                                } catch (e: Exception) { Log.e("YOLO_AI", "推理崩溃", e) }
                                finally {
                                    // 务必手动释放 OpenCV Mat 内存，否则会导致内存泄漏与崩溃
                                    fullMat.release(); leftMat.release(); rightMat.release()
                                    grayL.release(); grayR.release(); rectL.release(); rectR.release()
                                    disparity16S.release(); disparity32F.release()
                                    isInferencing = false // 解锁，允许处理下一帧
                                }
                            }.start()
                        }
                    } catch (e: Exception) {
                        Log.e("CameraPreview", "外层图像提取崩溃拦截", e)
                    } finally { isDrawing = false }
                }
            }
        })
    }

    /**
     * 语音合成播报函数
     * @param text 需要朗读的中文文本
     * @param isInterrupt 🌟 是否拥有最高打断特权（仅按键和大模型回复、跌倒报警允许为 true）
     */
    private fun speakOut(text: String, isInterrupt: Boolean = false) {
        if (isInterrupt) {
            // 🚨 强打断特权：立刻清空排队队列，切断当前废话强制插播 (QUEUE_FLUSH)
            tts.speak(text, TextToSpeech.QUEUE_FLUSH, null, "")
        } else {
            // 🚶 老老实实排队：追加到队列末尾，耐心等前面的说完 (QUEUE_ADD)
            tts.speak(text, TextToSpeech.QUEUE_ADD, null, "")
        }
    }

    /**
     * 从 assets 文件夹加载 TFLite 模型文件到内存映射中
     * @param modelName 模型文件名
     * @return 映射的 ByteBuffer
     */
    private fun loadModelFile(modelName: String): MappedByteBuffer {
        val fd = assets.openFd(modelName)
        return FileInputStream(fd.fileDescriptor).channel.map(FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength)
    }

    /**
     * 将 Bitmap 图片压缩并转换为 Base64 字符串格式
     * 用于通过 HTTP 接口将图片传送给云端大模型
     * @param bitmap 待转换的图像
     * @return Base64 编码的字符串
     */
    private fun bitmapToBase64(bitmap: Bitmap): String {
        val outputStream = ByteArrayOutputStream()
        bitmap.compress(Bitmap.CompressFormat.JPEG, 60, outputStream)
        return Base64.encodeToString(outputStream.toByteArray(), Base64.NO_WRAP)
    }

    /**
     * 封装 HTTP 请求，向云端 AutoDL 部署的 Qwen-VL 模型提问
     * @param prompt 用户的语音提问内容
     */
    private fun askLlm(prompt: String) {
        val currentFrame = latestBitmap ?: return
        Thread {
            try {
                // 🌟 核心修复：强制约束大模型的“话痨”属性！
                // 在用户真实提问前，强行植入一段极其严格的身份约束和字数限制指令。
                val strictPrompt = "你是一个为视障人士服务的导盲助手。请用极其简短、精炼的一句话回答，字数务必控制在20字以内，绝不能有废话！用户问：$prompt"

                // 构建 Qwen-VL 所需的 JSON Payload，传入加上了紧箍咒的 strictPrompt
                val json = JSONObject().apply {
                    put("model", MODEL_NAME); put("prompt", strictPrompt); put("stream", false)
                    put("images", JSONArray().apply { put(bitmapToBase64(currentFrame)) }) // 传入单张多模态图片
                }
                // 发送 POST 请求
                client.newCall(Request.Builder().url("$API_URL/api/generate").post(json.toString().toRequestBody("application/json; charset=utf-8".toMediaType())).build()).enqueue(object : Callback {
                    override fun onFailure(call: Call, e: IOException) {
                        speakOut("网络失败", isInterrupt = true)
                    }
                    override fun onResponse(call: Call, response: Response) {
                        response.use {
                            // 🌟 大模型返回结果，也是按键触发的一部分，拥有最高打断特权
                            if (it.isSuccessful) speakOut(JSONObject(it.body?.string() ?: "").getString("response"), isInterrupt = true)
                        }
                    }
                })
            } catch (e: Exception) {}
        }.start()
    }

    /**
     * 生命周期销毁方法
     * 释放线程池、相机服务、TTS 和跌倒检测传感器
     */
    override fun onDestroy() {
        super.onDestroy()
        locationTimer?.cancel() // 🌟 必须手动销毁定时器，防止内存泄漏和后台异常费电
        cameraExecutor.shutdown()
        mCameraClient?.closeCamera()
        if (::tts.isInitialized) tts.shutdown()
        if (::fallDetector.isInitialized) fallDetector.stop()
    }

    /**
     * 拦截手机实体按键事件 (用作无障碍交互入口)
     * 按下【音量减小】键触发系统语音识别
     */
    override fun onKeyDown(keyCode: Int, event: KeyEvent?): Boolean {
        if (keyCode == KeyEvent.KEYCODE_VOLUME_DOWN && event?.repeatCount == 0) {
            val currentTime = System.currentTimeMillis()
            if (currentTime - lastVolumeClickTime > 1500L) { // 1.5秒防连按控制
                lastVolumeClickTime = currentTime
                // 🌟 核心打断点：按键瞬间物理停止所有正在发音的语句！
                if (::tts.isInitialized) tts.stop()
                try {
                    // 拉起安卓自带的语音识别器
                    voiceLauncher.launch(Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
                        putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
                        putExtra(RecognizerIntent.EXTRA_LANGUAGE, Locale.CHINESE.toString())
                    })
                } catch (e: Exception) {}
            }
            return true // 表示事件已消费，系统不再降低音量
        }
        return super.onKeyDown(keyCode, event)
    }
}