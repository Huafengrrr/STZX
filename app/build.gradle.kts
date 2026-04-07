plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.android)
}

android {
    namespace = "com.example.myapplication1"

    // 🌟 修复语法错误：统一使用标准的 SDK 版本号
    compileSdk = 34

    // 告诉打包工具不要压缩 tflite 模型，否则 TFLite 引擎读取时会崩溃
    aaptOptions {
        noCompress.add("tflite")
    }

    defaultConfig {
        applicationId = "com.example.myapplication1"
        minSdk = 24
        targetSdk = 34 // 提升至 34 以满足现代安卓系统的安全与性能要求
        versionCode = 1
        versionName = "1.0"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"


    }

    buildTypes {
        release {
            // 🌟 瘦身绝招二：开启代码混淆与压缩
            isMinifyEnabled = true
            // 🌟 瘦身绝招三：开启无用资源移除（比如没用到的图片、布局文件）
            isShrinkResources = true

            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }

    kotlinOptions {
        jvmTarget = "11"
    }

    // 👇 🌟 在 android { } 的最后面加上这段“暴力剔除”代码
    packaging {
        jniLibs {
            // 物理拉黑：绝对不允许这四个架构的库打包进 APK！
            excludes.add("lib/x86/**")
            excludes.add("lib/x86_64/**")
            excludes.add("lib/armeabi-v7a/**")
            excludes.add("lib/armeabi/**")
        }
    }

    splits {
        abi {
            isEnable = true      // 开启分包
            reset()              // 清空所有默认架构
            include("arm64-v8a") // 只允许打包目前手机唯一主流的 64 位架构
            isUniversalApk = false // 坚决不打全架构混合包！
        }
    }

}

dependencies {
    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.appcompat)
    implementation(libs.material)
    implementation(libs.androidx.activity)
    implementation(libs.androidx.constraintlayout)
    testImplementation(libs.junit)
    androidTestImplementation(libs.androidx.junit)
    androidTestImplementation(libs.androidx.espresso.core)

    // OkHttp 用于后续把图片发给后端的 Python/Node.js 服务器
    implementation("com.squareup.okhttp3:okhttp:4.12.0")

    // TensorFlow Lite 核心库和辅助图像处理库
    implementation("org.tensorflow:tensorflow-lite:2.14.0")
    implementation("org.tensorflow:tensorflow-lite-support:0.4.4")

    // 🌟 原作者最新重写的 v3.x 架构！稳定，无底层报错！
    implementation("com.github.jiangdongguo.AndroidUSBCamera:libausbc:3.2.0")

    // 🌟 直接从云端拉取官方已经编译好的 OpenCV 终极包！彻底告别 NDK！
    implementation("org.opencv:opencv:4.10.0")

    // 🌟 添加官方新版 Splash Screen 开屏兼容库
    implementation("androidx.core:core-splashscreen:1.0.1")
}

