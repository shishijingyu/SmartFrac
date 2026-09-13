import { createRouter, createWebHistory } from 'vue-router'

const routes = [
  {
    path: '/',
    component: () => import('../layout/MainLayout.vue'),
    redirect: '/dashboard',
    children: [
      {
        path: 'dashboard',
        name: 'Dashboard',
        component: () => import('../views/Dashboard.vue'),
        meta: { title: '首页概览' }
      },
      {
        path: 'projects',
        name: 'Projects',
        component: () => import('../views/ProjectList.vue'),
        meta: { title: '项目管理' }
      },
      {
        path: 'cases',
        name: 'Cases',
        component: () => import('../views/CaseList.vue'),
        meta: { title: '算例管理' }
      },
      {
        path: 'tasks',
        name: 'Tasks',
        component: () => import('../views/TaskQueue.vue'),
        meta: { title: '任务队列' }
      },
      {
        path: 'models',
        name: 'Models',
        component: () => import('../views/ModelList.vue'),
        meta: { title: '模型管理' }
      },
      {
        path: 'results',
        name: 'Results',
        component: () => import('../views/ResultVisualization.vue'),
        meta: { title: '结果可视化' }
      }
    ]
  }
]

const router = createRouter({
  history: createWebHistory(),
  routes
})

export default router
