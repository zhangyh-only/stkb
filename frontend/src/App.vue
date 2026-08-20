<script setup lang="ts">
import { onMounted, ref } from 'vue'

type KnowledgeForm = {
  key: string
  name: string
  technology: string
  role: string
  status: 'configured' | 'capability_pending'
}

type Foundation = {
  stage: string
  positioning: string
  primary_store: string
  knowledge_forms: KnowledgeForm[]
}

const apiBaseUrl = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000/api'
const foundation = ref<Foundation | null>(null)
const error = ref('')
const loading = ref(true)

onMounted(async () => {
  try {
    const response = await fetch(`${apiBaseUrl}/foundation`)
    if (!response.ok) throw new Error(`HTTP ${response.status}`)
    foundation.value = await response.json()
  } catch (reason) {
    error.value = reason instanceof Error ? reason.message : '无法读取 API 状态'
  } finally {
    loading.value = false
  }
})
</script>

<template>
  <main class="shell">
    <header class="hero">
      <p class="eyebrow">STKB / VALIDATION LAB</p>
      <h1>方案要能运行，数据要能看见。</h1>
      <p class="summary">
        当前只提供工程底座。后续每个方案切片都通过 Python、Vue 和真实存储数据完成验证，不扩张成完整系统。
      </p>
    </header>

    <section class="status-bar">
      <div>
        <span>项目阶段</span>
        <strong>{{ foundation?.stage ?? '正在连接 API' }}</strong>
      </div>
      <div>
        <span>业务事实主存</span>
        <strong>{{ foundation?.primary_store ?? 'PostgreSQL' }}</strong>
      </div>
      <div>
        <span>API 状态</span>
        <strong :class="error ? 'is-error' : loading ? '' : 'is-ready'">
          {{ loading ? '连接中' : error || '可访问' }}
        </strong>
      </div>
    </section>

    <section>
      <div class="section-heading">
        <p>THREE KNOWLEDGE FORMS</p>
        <h2>三种知识形态</h2>
      </div>

      <div class="grid">
        <article v-for="(item, index) in foundation?.knowledge_forms ?? []" :key="item.key">
          <div class="index">0{{ index + 1 }}</div>
          <p class="technology">{{ item.technology }}</p>
          <h3>{{ item.name }}</h3>
          <p>{{ item.role }}</p>
          <span class="badge" :class="item.status">
            {{ item.status === 'configured' ? '底座已配置' : '能力待实现' }}
          </span>
        </article>
      </div>
    </section>

    <section class="next-step">
      <p>下一步不是补齐后台功能，而是选择一个真实方案问题，让同一份数据依次形成知识文件、向量投影和图投影，并在这里展示全过程。</p>
    </section>
  </main>
</template>
