'use strict';
const db = uniCloud.database();
const dbCmd = db.command;

// 生成 6 位随机数字配对码的辅助函数
function generatePairingCode() {
    return Math.floor(100000 + Math.random() * 900000).toString();
}

exports.main = async (event, context) => {
    const query = event.queryStringParameters || {};
    let body = {};

    // 尝试解析 POST 请求的 Body
    if (event.body) {
        try {
            body = typeof event.body === 'string' ? JSON.parse(event.body) : event.body;
        } catch (e) {
            return { success: false, msg: "JSON解析失败" };
        }
    }

    // 决定当前请求的操作类型（优先使用 body 中的 action）
    const action = body.action || query.action || "uploadLocation";

    // ==========================================
    // 🟢 模块 1：家属端 账号注册与登录
    // ==========================================
    if (action === 'register') {
        const { phone, password } = body;
        if (!phone || !password) return { success: false, msg: "手机号或密码不能为空" };

        const exist = await db.collection('stzx_users').where({ phone }).get();
        if (exist.data.length > 0) return { success: false, msg: "该手机号已注册" };

        await db.collection('stzx_users').add({
            phone: phone,
            password: password,
            bound_device_id: "",       // 兼容旧字段
            bound_device_ids: [],      // 🌟 新增：多设备列表
            create_time: Date.now()
        });
        return { success: true, msg: "注册成功" };
    }

    if (action === 'login') {
        const { phone, password } = body;
        const res = await db.collection('stzx_users').where({ phone, password }).get();
        if (res.data.length === 0) return { success: false, msg: "账号或密码错误" };

        const user = res.data[0];

        // 🌟 兼容处理：合并旧的 bound_device_id 和新的 bound_device_ids
        let deviceIds = Array.isArray(user.bound_device_ids) ? [...user.bound_device_ids] : [];
        if (user.bound_device_id && !deviceIds.includes(user.bound_device_id)) {
            deviceIds.push(user.bound_device_id);
        }
        // 过滤空字符串
        deviceIds = deviceIds.filter(id => id && id !== "");

        return {
            success: true,
            msg: "登录成功",
            bound_device_id: deviceIds.length > 0 ? deviceIds[0] : "",  // 兼容旧客户端
            bound_device_ids: deviceIds                                   // 🌟 新增：完整列表
        };
    }

    // ==========================================
    // 🟢 模块 1.5：家属端 获取设备列表
    // ==========================================
    if (action === 'getDeviceList') {
        const { phone } = body;
        if (!phone) return { success: false, msg: "缺少手机号" };

        const res = await db.collection('stzx_users').where({ phone }).get();
        if (res.data.length === 0) return { success: false, msg: "用户不存在" };

        const user = res.data[0];
        let deviceIds = Array.isArray(user.bound_device_ids) ? [...user.bound_device_ids] : [];
        if (user.bound_device_id && !deviceIds.includes(user.bound_device_id)) {
            deviceIds.push(user.bound_device_id);
        }
        deviceIds = deviceIds.filter(id => id && id !== "");

        return { success: true, deviceIds: deviceIds };
    }

    // ==========================================
    // 🟢 模块 1.6：家属端 解绑设备
    // ==========================================
    if (action === 'unbindDevice') {
        const { phone, deviceId } = body;
        if (!phone || !deviceId) return { success: false, msg: "缺少手机号或设备ID" };

        const res = await db.collection('stzx_users').where({ phone }).get();
        if (res.data.length === 0) return { success: false, msg: "用户不存在" };

        const user = res.data[0];
        let deviceIds = Array.isArray(user.bound_device_ids) ? [...user.bound_device_ids] : [];
        if (user.bound_device_id && !deviceIds.includes(user.bound_device_id)) {
            deviceIds.push(user.bound_device_id);
        }
        deviceIds = deviceIds.filter(id => id && id !== "" && id !== deviceId);

        const newPrimary = deviceIds.length > 0 ? deviceIds[0] : "";
        await db.collection('stzx_users').where({ phone }).update({
            bound_device_ids: deviceIds,
            bound_device_id: newPrimary
        });

        // 标记盲杖为未绑定
        await db.collection('stzx_devices').where({ device_id: deviceId }).update({
            is_bound: false
        }).catch(e => {});

        return { success: true, msg: "解绑成功", remaining_device_ids: deviceIds };
    }

    // ==========================================
    // 🟡 模块 2：盲杖端 静默生成配对码
    // ==========================================
    if (action === 'initDevice') {
        const { deviceId } = body;
        if (!deviceId) return { success: false, msg: "缺少设备硬件ID" };

        const exist = await db.collection('stzx_devices').where({ device_id: deviceId }).get();
        if (exist.data.length > 0) {
            return { success: true, pairingCode: exist.data[0].pairing_code, isBound: exist.data[0].is_bound };
        }

        const newCode = generatePairingCode();
        await db.collection('stzx_devices').add({
            device_id: deviceId,
            pairing_code: newCode,
            is_bound: false,
            create_time: Date.now()
        });
        return { success: true, pairingCode: newCode, isBound: false };
    }

    // ==========================================
    // 🟠 模块 3：家属端 输入配对码进行绑定
    // ==========================================
    if (action === 'bindDevice') {
        const { phone, pairingCode } = body;
        if (!phone || !pairingCode) return { success: false, msg: "缺少手机号或配对码" };

        const deviceRes = await db.collection('stzx_devices').where({ pairing_code: pairingCode }).get();
        if (deviceRes.data.length === 0) return { success: false, msg: "配对码无效，请检查盲杖播报" };

        const deviceId = deviceRes.data[0].device_id;

        // 🌟 追加到多设备列表（去重）
        const userRes = await db.collection('stzx_users').where({ phone }).get();
        if (userRes.data.length === 0) return { success: false, msg: "用户不存在" };

        const user = userRes.data[0];
        let deviceIds = Array.isArray(user.bound_device_ids) ? [...user.bound_device_ids] : [];
        if (user.bound_device_id && !deviceIds.includes(user.bound_device_id)) {
            deviceIds.push(user.bound_device_id);
        }
        deviceIds = deviceIds.filter(id => id && id !== "");
        if (!deviceIds.includes(deviceId)) {
            deviceIds.push(deviceId);
        }

        await db.collection('stzx_users').where({ phone }).update({
            bound_device_ids: deviceIds,
            bound_device_id: deviceIds[0]   // 主设备保持第一个
        });

        await db.collection('stzx_devices').where({ device_id: deviceId }).update({
            is_bound: true
        });

        return { success: true, msg: "盲杖绑定成功！", deviceId: deviceId };
    }

    // ==========================================
    // 🔵 模块 4：盲杖端 上传位置数据
    // ==========================================
    if (action === 'uploadLocation') {
        if (!body.longitude || !body.deviceId) return { success: false, msg: "缺少经纬度或设备ID" };

        const now = Date.now();
        await db.collection('location_data').add({
            deviceId: body.deviceId,
            longitude: body.longitude,
            latitude: body.latitude,
            status: body.status || "Normal",
            create_time: now
        });

        const fiveDaysAgo = now - (5 * 24 * 60 * 60 * 1000);
        db.collection('location_data').where({ create_time: dbCmd.lt(fiveDaysAgo) }).remove().catch(e => {});

        return { success: true, msg: "写入成功" };
    }

    // ==========================================
    // 🟣 模块 5：家属端 拉取位置数据
    // ==========================================

    // 🟣-1: POST 方式 —— fetchLocation action
    if (action === 'fetchLocation') {
        const { deviceId } = body;
        if (!deviceId) return { success: false, msg: "缺少设备ID" };

        const res = await db.collection('location_data').where({
            deviceId: deviceId
        }).orderBy("create_time", "desc").limit(1).get();

        return { success: true, data: res.data[0] || null };
    }

    // 🟣-2: GET 方式 —— 通过 query 参数（history 查询历史）
    if (query.action === 'history' && query.date) {
        const targetDeviceId = query.deviceId;
        if (!targetDeviceId) return { success: false, msg: "未提供设备ID" };

        const startOfDay = new Date(query.date + 'T00:00:00+08:00').getTime();
        const endOfDay = startOfDay + 24 * 60 * 60 * 1000;
        const res = await db.collection('location_data').where({
            deviceId: targetDeviceId,
            create_time: dbCmd.gte(startOfDay).and(dbCmd.lt(endOfDay))
        }).orderBy("create_time", "asc").limit(1000).get();
        return { success: true, mode: "history", data: res.data };
    }

    // 🟣-3: GET 方式 —— 查询最新一条位置（实时模式）
    if (query.deviceId) {
        const res = await db.collection('location_data').where({
            deviceId: query.deviceId
        }).orderBy("create_time", "desc").limit(1).get();
        return { success: true, mode: "realtime", data: res.data[0] || null };
    }

    return { success: false, msg: "不支持的请求方式" };
};
