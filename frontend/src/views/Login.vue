<template>
  <div class="login-container">
    <!-- 背景装饰 -->
    <div class="background-decoration">
      <div class="decoration-circle circle-1"></div>
      <div class="decoration-circle circle-2"></div>
      <div class="decoration-circle circle-3"></div>
    </div>

    <!-- 登录卡片 -->
    <div class="login-card">
      <div class="login-header">
        <h1 class="login-title">
          <span class="icon">🌍</span>
          智能旅行助手
        </h1>
        <p class="login-subtitle">登录您的账户，开启智能旅程</p>
      </div>

      <!-- 登录表单 -->
      <el-tabs v-model="activeTab" class="login-tabs">
        <!-- 登录标签页 -->
        <el-tab-pane label="登录" name="login">
          <el-form :model="loginForm" :rules="loginRules" ref="loginFormRef" class="login-form">
            <el-form-item prop="username">
              <el-input
                v-model="loginForm.username"
                placeholder="请输入账号"
                prefix-icon="User"
                size="large"
              />
            </el-form-item>
            
            <el-form-item prop="password">
              <el-input
                v-model="loginForm.password"
                placeholder="请输入密码"
                prefix-icon="Lock"
                size="large"
                type="password"
                show-password
                @keyup.enter="handleLogin"
              />
            </el-form-item>
            
            <el-form-item>
              <el-button
                type="primary"
                size="large"
                :loading="loginLoading"
                @click="handleLogin"
                class="submit-button"
              >
                {{ loginLoading ? '登录中...' : '登录' }}
              </el-button>
            </el-form-item>
          </el-form>
        </el-tab-pane>

        <!-- 注册标签页 -->
        <el-tab-pane label="注册" name="register">
          <el-form :model="registerForm" :rules="registerRules" ref="registerFormRef" class="login-form">
            <el-form-item prop="username">
              <el-input
                v-model="registerForm.username"
                placeholder="请输入账号"
                prefix-icon="User"
                size="large"
              />
            </el-form-item>
            
            <el-form-item prop="password">
              <el-input
                v-model="registerForm.password"
                placeholder="请输入密码（至少6位）"
                prefix-icon="Lock"
                size="large"
                type="password"
                show-password
              />
            </el-form-item>
            
            <el-form-item>
              <el-button
                type="primary"
                size="large"
                :loading="registerLoading"
                @click="handleRegister"
                class="submit-button"
              >
                {{ registerLoading ? '注册中...' : '注册' }}
              </el-button>
            </el-form-item>
          </el-form>
        </el-tab-pane>
      </el-tabs>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, reactive } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import { User, Lock } from '@element-plus/icons-vue'
import { authApi } from '@/services/api'
import type { LoginRequest, RegisterRequest } from '@/types'
import type { FormInstance } from 'element-plus'
import { useAuthStore } from '@/stores/auth'

const router = useRouter()
const authStore = useAuthStore()

// 表单引用
const loginFormRef = ref<FormInstance>()
const registerFormRef = ref<FormInstance>()

// 活动标签页
const activeTab = ref('login')

// 加载状态
const loginLoading = ref(false)
const registerLoading = ref(false)

// 登录表单
const loginForm = reactive<LoginRequest>({
  username: '',
  password: ''
})

// 注册表单
const registerForm = reactive<RegisterRequest>({
  username: '',
  password: ''
})

// 登录表单验证规则
const loginRules = {
  username: [
    { required: true, message: '请输入账号', trigger: 'blur' }
  ],
  password: [
    { required: true, message: '请输入密码', trigger: 'blur' },
    { min: 6, message: '密码长度不能少于6位', trigger: ['blur', 'change'] },
    { max: 50, message: '密码长度不能超过50位', trigger: ['blur', 'change'] }
  ]
}

// 注册表单验证规则
const registerRules = {
  username: [
    { required: true, message: '请输入账号', trigger: 'blur' },
    { min: 2, message: '账号长度不能少于2位', trigger: ['blur', 'change'] },
    { max: 20, message: '账号长度不能超过20位', trigger: ['blur', 'change'] }
  ],
  password: [
    { required: true, message: '请输入密码', trigger: 'blur' },
    { min: 6, message: '密码长度不能少于6位', trigger: ['blur', 'change'] },
    { max: 50, message: '密码长度不能超过50位', trigger: ['blur', 'change'] }
  ]
}

// 处理登录
const handleLogin = async () => {
  if (!loginFormRef.value) return
  
  const valid = await loginFormRef.value.validate().catch(() => false)
  if (!valid) return

  loginLoading.value = true
  
  try {
    const response = await authApi.login(loginForm)
    
    // 保存认证信息到localStorage
    localStorage.setItem('access_token', response.access_token)
    localStorage.setItem('user_info', JSON.stringify(response.user))
    
    // 更新认证状态
    authStore.setAuth(response.access_token, response.user)
    
    ElMessage.success('登录成功！')
    
    // 跳转到首页
    router.push('/')
  } catch (error: any) {
    ElMessage.error(error.message || '登录失败，请检查用户名和密码')
  } finally {
    loginLoading.value = false
  }
}

// 处理注册
const handleRegister = async () => {
  if (!registerFormRef.value) return
  
  const valid = await registerFormRef.value.validate().catch(() => false)
  if (!valid) return

  registerLoading.value = true
  
  try {
    const response = await authApi.register(registerForm)
    
    // 保存认证信息到localStorage
    localStorage.setItem('access_token', response.access_token)
    localStorage.setItem('user_info', JSON.stringify(response.user))
    
    // 更新认证状态
    authStore.setAuth(response.access_token, response.user)
    
    ElMessage.success('注册成功！')
    
    // 跳转到首页
    router.push('/')
  } catch (error: any) {
    ElMessage.error(error.message || '注册失败，请重试')
  } finally {
    registerLoading.value = false
  }
}
</script>

<style scoped lang="scss">
.login-container {
  position: relative;
  min-height: 100vh;
  padding: 24px;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  display: flex;
  align-items: center;
  justify-content: center;
  overflow: hidden;

  // 背景装饰
  .background-decoration {
    position: absolute;
    width: 100%;
    height: 100%;
    overflow: hidden;
    pointer-events: none;

    .decoration-circle {
      position: absolute;
      border-radius: 50%;
      background: rgba(255, 255, 255, 0.1);
      animation: float 20s infinite;

      &.circle-1 {
        width: 300px;
        height: 300px;
        top: -150px;
        left: -150px;
        animation-delay: 0s;
      }

      &.circle-2 {
        width: 200px;
        height: 200px;
        top: 50%;
        right: -100px;
        animation-delay: 5s;
      }

      &.circle-3 {
        width: 400px;
        height: 400px;
        bottom: -200px;
        left: 30%;
        animation-delay: 10s;
      }
    }
  }

  // 登录卡片
  .login-card {
    position: relative;
    width: 100%;
    max-width: 400px;
    border-radius: 16px;
    box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
    background: rgba(255, 255, 255, 0.95);
    backdrop-filter: blur(10px);
    padding: 40px;
    z-index: 1;
    animation: fadeInUp 0.8s ease;

    .login-header {
      text-align: center;
      margin-bottom: 30px;

      .login-title {
        margin: 0 0 20px 0;
        font-size: 28px;
        font-weight: 700;
        color: #333;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);

        .icon {
          display: inline-block;
          animation: rotate 3s ease-in-out infinite;
        }
      }

      .login-subtitle {
        margin: 0;
        font-size: 16px;
        color: #666;
        opacity: 0.8;
      }
    }

    .login-tabs {
      margin-bottom: 20px;

      :deep(.el-tabs__header) {
        margin: 0 0 20px 0;
      }

      :deep(.el-tabs__nav-wrap) {
        &::after {
          background-color: #e4e7ed;
        }
      }
    }

    .login-form {
      :deep(.el-form-item) {
        margin-bottom: 20px;
      }

      :deep(.el-input__wrapper) {
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        border-radius: 8px;
        transition: all 0.3s;

        &:hover {
          box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        }
      }

      .submit-button {
        width: 100%;
        height: 50px;
        font-size: 16px;
        font-weight: 600;
        border-radius: 8px;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
        transition: all 0.3s;

        &:hover {
          transform: translateY(-2px);
          box-shadow: 0 6px 16px rgba(102, 126, 234, 0.5);
        }
      }
    }
  }
}

// 动画
@keyframes float {
  0%, 100% {
    transform: translateY(0) rotate(0deg);
  }
  50% {
    transform: translateY(-20px) rotate(180deg);
  }
}

@keyframes rotate {
  from {
    transform: rotate(0deg);
  }
  to {
    transform: rotate(360deg);
  }
}

@keyframes fadeInUp {
  from {
    opacity: 0;
    transform: translateY(30px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

// 响应式设计
@media (max-width: 480px) {
  .login-container {
    align-items: flex-start;
    padding: 24px 12px;
    overflow: auto;

    .login-card {
      max-width: 100%;
      padding: 24px 18px;

      .login-header {
        margin-bottom: 24px;

        .login-title {
          font-size: 24px;
          line-height: 1.25;
        }

        .login-subtitle {
          font-size: 14px;
          line-height: 1.6;
        }
      }
    }
  }
}
</style>
