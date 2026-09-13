<template>
  <div>
    <el-row :gutter="16">
      <el-col :span="6" v-for="card in statCards" :key="card.title">
        <el-card shadow="hover">
          <div style="font-size: 13px; color: #909399;">{{ card.title }}</div>
          <div style="font-size: 28px; font-weight: 600; margin-top: 8px; color: #303133;">{{ card.value }}</div>
        </el-card>
      </el-col>
    </el-row>

    <el-card style="margin-top: 16px;" shadow="never">
      <template #header>
        <span>系统状态</span>
      </template>
      <el-descriptions :column="2" border>
        <el-descriptions-item label="Java 后端">
          <el-tag type="success" size="small">运行中</el-tag>
        </el-descriptions-item>
        <el-descriptions-item label="Python 内核">
          <el-tag type="warning" size="small">待启动</el-tag>
        </el-descriptions-item>
        <el-descriptions-item label="MySQL">
          <el-tag type="warning" size="small">待连接</el-tag>
        </el-descriptions-item>
        <el-descriptions-item label="RabbitMQ">
          <el-tag type="warning" size="small">待连接</el-tag>
        </el-descriptions-item>
      </el-descriptions>
    </el-card>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { projectApi, taskApi } from '../api'

const statCards = ref([
  { title: '项目数', value: '-' },
  { title: '算例数', value: '-' },
  { title: '运行中任务', value: '-' },
  { title: '已完成任务', value: '-' }
])

onMounted(async () => {
  try {
    const [projects, runningTasks, completedTasks] = await Promise.all([
      projectApi.list(),
      taskApi.list('running'),
      taskApi.list('completed')
    ])
    statCards.value = [
      { title: '项目数', value: projects.data?.length || 0 },
      { title: '算例数', value: '-' },
      { title: '运行中任务', value: runningTasks.data?.length || 0 },
      { title: '已完成任务', value: completedTasks.data?.length || 0 }
    ]
  } catch (e) {
    console.error('加载统计数据失败', e)
  }
})
</script>
