package com.example.myapplication1

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.util.AttributeSet
import android.view.Choreographer
import android.view.View

/**
 * 带有运动预测 (Velocity Tracking) 的目标边界框数据类
 */
data class BoundingBox(
    var left: Float,
    var top: Float,
    var right: Float,
    var bottom: Float,
    val label: String,
    val score: Float
) {
    // 🌟 新增：由底层 AI 算出的目标在屏幕上的移动速度 (像素/秒)
    var vx: Float = 0f
    var vy: Float = 0f
}

/**
 * 支持 60FPS 平滑补帧动画的覆盖视图
 * 彻底解决 AI 帧率低导致的画框滞后和抖动问题
 */
class OverlayView(context: Context?, attrs: AttributeSet?) : View(context, attrs), Choreographer.FrameCallback {

    private val boxPaint = Paint().apply {
        color = Color.RED
        style = Paint.Style.STROKE
        strokeWidth = 6f
        isAntiAlias = true
    }

    private val textBackgroundPaint = Paint().apply {
        color = Color.parseColor("#80000000") // 半透明黑色背景，让文字更清晰
        style = Paint.Style.FILL
    }

    private val textPaint = Paint().apply {
        color = Color.WHITE
        textSize = 40f
        isAntiAlias = true
        typeface = android.graphics.Typeface.DEFAULT_BOLD
    }

    // AI 传过来的真实检测结果 (可能 1 秒只有 4 次更新)
    private var results = mutableListOf<BoundingBox>()

    // 🌟 新增：用于在 UI 层补帧的动画内部状态
    private var animatedResults = mutableListOf<BoundingBox>()
    private var lastUpdateTime = 0L
    private var interpolationEnabled = false

    /**
     * 接收来自 YOLO 推理线程的数据更新
     */
    fun setResults(newResults: List<BoundingBox>) {
        results.clear()
        results.addAll(newResults)

        // 每次收到真实数据，重置我们的动画初始帧，纠正累计的预测偏差
        animatedResults.clear()
        for (box in newResults) {
            animatedResults.add(box.copy())
        }

        lastUpdateTime = System.currentTimeMillis()

        if (!interpolationEnabled) {
            invalidate() // 如果没开动画，就老实刷新
        }
    }

    /**
     * 开启或关闭 60FPS 运动补帧插值动画
     */
    fun enableInterpolation(enable: Boolean) {
        if (enable && !interpolationEnabled) {
            interpolationEnabled = true
            Choreographer.getInstance().postFrameCallback(this)
        } else if (!enable && interpolationEnabled) {
            interpolationEnabled = false
            Choreographer.getInstance().removeFrameCallback(this)
        }
    }

    /**
     * 🌟 核心魔法：系统底层的 60FPS 渲染回调
     * 在 AI 推理“卡住”的这几百毫秒里，UI 并不闲着，而是根据目标的 vx, vy 惯性继续推着框走！
     */
    override fun doFrame(frameTimeNanos: Long) {
        if (interpolationEnabled && animatedResults.isNotEmpty()) {
            val currentTime = System.currentTimeMillis()
            // 计算距离上一次收到真实 AI 数据过去了多少秒
            val dt = (currentTime - lastUpdateTime) / 1000f

            // 安全机制：如果太久没收到新数据（比如 >0.5秒），说明物体可能消失了或者卡顿严重，停止插值以防框飞出屏幕外
            if (dt < 0.5f) {
                // 基于初始状态和速度，预测此刻的框应该在哪里
                for (i in animatedResults.indices) {
                    if (i < results.size) {
                        val realBox = results[i]
                        val animBox = animatedResults[i]

                        // 惯性平移 (预测距离 = 速度 * 时间)
                        val dx = realBox.vx * dt
                        val dy = realBox.vy * dt

                        animBox.left = realBox.left + dx
                        animBox.right = realBox.right + dx
                        animBox.top = realBox.top + dy
                        animBox.bottom = realBox.bottom + dy
                    }
                }
                invalidate() // 强制重新绘制预测的新位置
            }
        }

        // 循环注册回调，保持动画运行
        if (interpolationEnabled) {
            Choreographer.getInstance().postFrameCallback(this)
        }
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)

        // 决定画哪个列表：如果开了插值就画 animatedResults，否则画原生的 results
        val listToDraw = if (interpolationEnabled) animatedResults else results

        for (result in listToDraw) {
            val rect = RectF(result.left, result.top, result.right, result.bottom)

            // 1. 画红框
            canvas.drawRect(rect, boxPaint)

            // 2. 画标签文字和半透明背景条
            val text = result.label
            val textWidth = textPaint.measureText(text)
            val fontMetrics = textPaint.fontMetrics
            val textHeight = fontMetrics.bottom - fontMetrics.top

            // 将文字悬浮在框的左上角外部
            val bgRect = RectF(
                result.left,
                result.top - textHeight - 8f,
                result.left + textWidth + 16f,
                result.top
            )

            canvas.drawRect(bgRect, textBackgroundPaint)
            canvas.drawText(text, result.left + 8f, result.top - 8f - fontMetrics.bottom, textPaint)
        }
    }
}