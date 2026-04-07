package com.example.myapplication1

import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import kotlin.math.sqrt

// 🌟 独立的跌倒检测引擎
class FallDetector(context: Context, private val onFallDetected: () -> Unit) : SensorEventListener {
    private val sensorManager = context.getSystemService(Context.SENSOR_SERVICE) as SensorManager
    private val accelerometer = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)

    // ================= 物理特征阈值配置 =================
    // 正常重力加速度约为 9.8 m/s²
    private val FREE_FALL_THRESHOLD = 4.0f // 失重阈值：小于这个值认为正在自由落体
    private val IMPACT_THRESHOLD = 20.0f   // 撞击阈值：大于这个值认为砸到了地面

    private var isFreeFalling = false
    private var freeFallTime = 0L

    fun start() {
        accelerometer?.let {
            // 使用 SENSOR_DELAY_GAME 级别，采样率更高，适合捕捉瞬间的摔倒撞击
            sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_GAME)
        }
    }

    fun stop() {
        sensorManager.unregisterListener(this)
    }

    override fun onSensorChanged(event: SensorEvent?) {
        if (event?.sensor?.type == Sensor.TYPE_ACCELEROMETER) {
            val x = event.values[0]
            val y = event.values[1]
            val z = event.values[2]

            // 1. 计算三轴合加速度 A = √(x² + y² + z²)
            val acceleration = sqrt((x * x + y * y + z * z).toDouble()).toFloat()

            // 2. 捕捉【失重期】：身体正在下坠，加速度急剧减小
            if (acceleration < FREE_FALL_THRESHOLD) {
                if (!isFreeFalling) {
                    isFreeFalling = true
                    freeFallTime = System.currentTimeMillis()
                }
            }

            // 3. 捕捉【撞击期】：失重之后，紧接着出现巨大的加速度峰值
            if (isFreeFalling && acceleration > IMPACT_THRESHOLD) {
                val timeDifference = System.currentTimeMillis() - freeFallTime

                // 摔倒过程极快，通常从下坠到砸地在 1 秒（1000毫秒）内完成
                if (timeDifference < 1000L) {
                    onFallDetected() // 🌟 触发警报！
                    isFreeFalling = false // 重置状态
                }
            }

            // 4. 超时重置机制：如果失重后过了很久都没撞击（比如手机被轻轻抛起又接住），取消判定
            if (isFreeFalling && System.currentTimeMillis() - freeFallTime > 1500L) {
                isFreeFalling = false
            }
        }
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}
}