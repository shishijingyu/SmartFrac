<template>
  <div>
    <el-card shadow="never">
      <template #header>
        <div style="display: flex; justify-content: space-between; align-items: center;">
          <span>任务队列</span>
          <el-radio-group v-model="filterStatus" size="small" @change="loadTasks">
            <el-radio-button value="">全部</el-radio-button>
            <el-radio-button value="pending">排队中</el-radio-button>
            <el-radio-button value="running">运行中</el-radio-button>
            <el-radio-button value="completed">已完成</el-radio-button>
            <el-radio-button value="failed">失败</el-radio-button>
          </el-radio-group>
        </div>
      </template>
      <el-table :data="tasks" stripe v-loading="loading">
        <el-table-column prop="taskNo" label="任务编号" width="180" />
        <el-table-column prop="taskType" label="类型" width="120">
          <template #default="{ row }">
            <el-tag size="small">{{ row.taskType }}</el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="status" label="状态" width="100">
          <template #default="{ row }">
            <el-tag :type="statusType(row.status)" size="small">{{ row.status }}</el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="progress" label="进度" width="150">
          <template #default="{ row }">
            <el-progress :percentage="row.progress || 0" :status="row.status === 'failed' ? 'exception' : ''" />
          </template>
        </el-table-column>
        <el-table-column prop="createTime" label="创建时间" width="180" />
        <el-table-column label="操作" width="100">
          <template #default="{ row }">
            <el-button size="small" type="danger" v-if="row.status === 'pending' || row.status === 'running'"
              @click="handleCancel(row.id)">取消</el-button>
          </template>
        </el-table-column>
      </el-table>
    </el-card>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { taskApi } from '../api'
import { ElMessage } from 'element-plus'

const tasks = ref([])
const loading = ref(false)
const filterStatus = ref('')

const loadTasks = async () => {
  loading.value = true
  try {
    const res = await taskApi.list(filterStatus.value || null)
    tasks.value = res.data || []
  } catch (e) {
    ElMessage.error('加载任务列表失败')
  } finally {
    loading.value = false
  }
}

const statusType = (s) => {
  const map = { pending: 'info', running: 'warning', completed: 'success', failed: 'danger', cancelled: 'info' }
  return map[s] || 'info'
}

const handleCancel = async (id) => {
  try {
    await taskApi.cancel(id)
    ElMessage.success('任务已取消')
    loadTasks()
  } catch (e) {
    ElMessage.error('取消失败')
  }
}

onMounted(loadTasks)
</script>
