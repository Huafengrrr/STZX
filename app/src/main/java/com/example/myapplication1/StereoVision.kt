package com.example.myapplication1

import org.opencv.calib3d.Calib3d
import org.opencv.calib3d.StereoSGBM
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Size

/**
 * 双目立体视觉工具类
 * 包含：相机标定参数、立体校正、SGBM 视差计算、WGS84 转 GCJ02 坐标转换、IoU 计算
 */
object StereoVision {

    // ===== OpenCV 双目相机标定参数 =====
    val cameraMatrixL by lazy {
        Mat(3, 3, CvType.CV_64F).apply {
            put(0, 0, 328.26391515, 0.0, 327.21824491)
            put(1, 0, 0.0, 328.38426787, 235.88575854)
            put(2, 0, 0.0, 0.0, 1.0)
        }
    }
    val distCoeffL by lazy {
        Mat(5, 1, CvType.CV_64F).apply {
            put(0, 0, 0.0706293, -0.0739329, -0.0660674, 0.0, 0.0)
        }
    }
    val cameraMatrixR by lazy {
        Mat(3, 3, CvType.CV_64F).apply {
            put(0, 0, 329.07585782, 0.0, 312.23129337)
            put(1, 0, 0.0, 329.28633065, 237.06551395)
            put(2, 0, 0.0, 0.0, 1.0)
        }
    }
    val distCoeffR by lazy {
        Mat(5, 1, CvType.CV_64F).apply {
            put(0, 0, 0.0767674, -0.0996155, 0.0, 0.0, 0.0)
        }
    }
    val Rot by lazy {
        Mat(3, 3, CvType.CV_64F).apply {
            put(0, 0, 0.99996649, -0.00016997, 0.00818365)
            put(1, 0, 0.00017301, 0.99999991, -0.00037007)
            put(2, 0, -0.00818358, 0.00037147, 0.99996644)
        }
    }
    val Trans by lazy {
        Mat(3, 1, CvType.CV_64F).apply {
            put(0, 0, -60.992415, -0.048313, -0.143001)
        }
    }

    // 立体校正映射矩阵缓存
    val Rl by lazy { Mat() }
    val Rr by lazy { Mat() }
    val Pl by lazy { Mat() }
    val Pr by lazy { Mat() }
    val Q by lazy { Mat() }
    val mapLx by lazy { Mat() }
    val mapLy by lazy { Mat() }
    val mapRx by lazy { Mat() }
    val mapRy by lazy { Mat() }

    // SGBM 半全局块匹配算法配置
    val sgbm by lazy {
        val blockWindowSize = 11
        StereoSGBM.create(0, 64, blockWindowSize).apply {
            setP1(8 * 1 * blockWindowSize * blockWindowSize)
            setP2(32 * 1 * blockWindowSize * blockWindowSize)
            setPreFilterCap(63)
            setMode(StereoSGBM.MODE_SGBM_3WAY)
            setUniquenessRatio(10)
            setSpeckleWindowSize(100)
            setSpeckleRange(32)
            setDisp12MaxDiff(1)
        }
    }

    /**
     * 初始化双目摄像头的立体校正映射矩阵
     */
    fun initStereoRectify() {
        val imageSize = Size(640.0, 480.0)
        Calib3d.stereoRectify(
            cameraMatrixL, distCoeffL, cameraMatrixR, distCoeffR,
            imageSize, Rot, Trans, Rl, Rr, Pl, Pr, Q,
            Calib3d.CALIB_ZERO_DISPARITY, 0.0, imageSize, null, null
        )
        Calib3d.initUndistortRectifyMap(cameraMatrixL, distCoeffL, Rl, Pl, imageSize, CvType.CV_32FC1, mapLx, mapLy)
        Calib3d.initUndistortRectifyMap(cameraMatrixR, distCoeffR, Rr, Pr, imageSize, CvType.CV_32FC1, mapRx, mapRy)
    }

    /**
     * 计算两个边界框的交并比 (IoU)，用于 NMS 和物体追踪
     */
    fun calculateIoU(box1: MainActivity.Detection, box2: MainActivity.Detection): Float {
        val x1 = maxOf(box1.cx - box1.w / 2f, box2.cx - box2.w / 2f)
        val y1 = maxOf(box1.cy - box1.h / 2f, box2.cy - box2.h / 2f)
        val x2 = minOf(box1.cx + box1.w / 2f, box2.cx + box2.w / 2f)
        val y2 = minOf(box1.cy + box1.h / 2f, box2.cy + box2.h / 2f)
        val intersectionArea = maxOf(0f, x2 - x1) * maxOf(0f, y2 - y1)
        val box1Area = box1.w * box1.h
        val box2Area = box2.w * box2.h
        return intersectionArea / (box1Area + box2Area - intersectionArea)
    }

    // ===== WGS84 转 GCJ02 坐标转换算法 =====

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
     * 将 GPS 坐标 (WGS84) 转换为高德地图火星坐标系 (GCJ02)
     * 彻底消灭国内 Android 手机定位和高德地图之间的偏移问题
     */
    fun wgs84ToGcj02(lon: Double, lat: Double): Pair<Double, Double> {
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
}
