pluginManagement {
    repositories {
        google {
            content {
                includeGroupByRegex("com\\.android.*")
                includeGroupByRegex("com\\.google.*")
                includeGroupByRegex("androidx.*")
            }
        }
        mavenCentral()
        gradlePluginPortal()
    }
}
dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories {
        google()
        mavenCentral()
        // 👇 新增这行，允许下载 Github 上的开源库
        maven { url = uri("https://jitpack.io") }
        // 🌟 2. 新增阿里云公共代理仓库（非常重要，能解决 90% 的包找不到问题）
        maven { url = uri("https://maven.aliyun.com/repository/public" ) }

        // 🌟 3. 新增这个特定仓库，专门解决 immersionbar 和 webpdecoder 的下载
        maven { url = uri("https://oss.sonatype.org/content/repositories/snapshots/") }
    }
}

rootProject.name = "My Application1"
include(":app")

