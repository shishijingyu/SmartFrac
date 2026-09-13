import request from './request'

export const projectApi = {
  list: () => request.get('/projects'),
  getById: (id) => request.get(`/projects/${id}`),
  create: (data) => request.post('/projects', data),
  update: (id, data) => request.put(`/projects/${id}`, data),
  delete: (id) => request.delete(`/projects/${id}`)
}

export const caseApi = {
  list: (projectId) => request.get('/cases', { params: { projectId } }),
  getById: (id) => request.get(`/cases/${id}`),
  create: (data) => request.post('/cases', data),
  update: (id, data) => request.put(`/cases/${id}`, data),
  delete: (id) => request.delete(`/cases/${id}`)
}

export const taskApi = {
  list: (status) => request.get('/tasks', { params: { status } }),
  getById: (id) => request.get(`/tasks/${id}`),
  submit: (data) => request.post('/tasks', data),
  cancel: (id) => request.post(`/tasks/${id}/cancel`)
}
