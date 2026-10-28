package com.example.myapplication1

import android.content.Context
import android.os.Handler
import android.os.Looper
import android.provider.Settings
import android.speech.tts.TextToSpeech
import android.speech.tts.UtteranceProgressListener
import android.util.Log
import okhttp3.Call
import okhttp3.Callback
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import okhttp3.Response
import org.json.JSONObject
import java.io.IOException

/**
 * 设备绑定管理器
 * 负责向云端申请配对码、播报配对码、轮询绑定状态
 */
class DeviceBindingManager(
    private val context: Context,
    private val tts: TextToSpeech,
    private val client: OkHttpClient,
    private val uniCloudUrl: String,
    private val onBoundSuccess: () -> Unit
) {
    companion object {
        private const val TAG = "DeviceBinding"
    }

    var deviceId: String = ""
        private set
    var isDeviceBound: Boolean = false
        private set

    private var isPairingCodeSpeaking = false

    /**
     * 读取硬件 ID 并向云端初始化设备，获取配对码后播报
     */
    fun init() {
        deviceId = Settings.Secure.getString(context.contentResolver, Settings.Secure.ANDROID_ID)

        val json = JSONObject().apply {
            put("action", "initDevice")
            put("deviceId", deviceId)
        }

        val request = Request.Builder()
            .url(uniCloudUrl)
            .post(json.toString().toRequestBody("application/json; charset=utf-8".toMediaType()))
            .build()

        client.newCall(request).enqueue(object : Callback {
            override fun onFailure(call: Call, e: IOException) {
                Log.e(TAG, "获取配对码网络失败", e)
                // 网络失败 10 秒后重试
                Handler(Looper.getMainLooper()).postDelayed({ init() }, 10000)
            }

            override fun onResponse(call: Call, response: Response) {
                val resStr = response.body?.string()
                try {
                    val resJson = JSONObject(resStr ?: "")
                    if (!resJson.getBoolean("success")) return

                    val isBound = resJson.getBoolean("isBound")
                    val code = resJson.getString("pairingCode")

                    if (!isBound) {
                        // 把配对码数字用双空格隔开，防止 TTS 读成十几万
                        val spacedCode = code.toCharArray().joinToString("  ")
                        val textToSpeak = "欢迎使用声途智行。您的盲杖配对码是：$spacedCode。请家属在亲友端输入绑定。"

                        Handler(Looper.getMainLooper()).post {
                            if (!isPairingCodeSpeaking) {
                                isPairingCodeSpeaking = true
                                // 播完后再继续轮询，杜绝播到一半被打断重来
                                tts.setOnUtteranceProgressListener(object : UtteranceProgressListener() {
                                    override fun onStart(utteranceId: String?) {}
                                    override fun onDone(utteranceId: String?) {
                                        isPairingCodeSpeaking = false
                                        if (!isDeviceBound) {
                                            Handler(Looper.getMainLooper()).postDelayed({ init() }, 3000)
                                        }
                                    }
                                    override fun onError(utteranceId: String?) {
                                        isPairingCodeSpeaking = false
                                    }
                                })
                                tts.speak(textToSpeak, TextToSpeech.QUEUE_FLUSH, null, "PAIRING_CODE")
                            }
                            // 正在播报中：轮询交由 onDone 回调负责，不重复打断
                        }
                    } else {
                        // 发现已被绑定
                        if (!isDeviceBound) {
                            Handler(Looper.getMainLooper()).post {
                                tts.speak("盲杖绑定成功！开始为您导航。", TextToSpeech.QUEUE_FLUSH, null, "")
                            }
                            isDeviceBound = true
                            onBoundSuccess()
                        }
                    }
                } catch (e: Exception) {
                    Log.e(TAG, "解析配对数据失败", e)
                }
            }
        })
    }
}
