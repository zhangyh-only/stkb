const API_PATH_PREFIX = '/api'

export function resolveApiBaseUrl(configuredValue?: string): string {
  const configuredBaseUrl = configuredValue?.trim().replace(/\/+$/, '')
  if (!configuredBaseUrl) return API_PATH_PREFIX
  return configuredBaseUrl.endsWith(API_PATH_PREFIX)
    ? configuredBaseUrl
    : `${configuredBaseUrl}${API_PATH_PREFIX}`
}
