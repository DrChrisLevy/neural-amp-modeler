import type { Capture, Job } from "./types"

export class ApiError extends Error {
  status: number
  constructor(message: string, status: number) {
    super(message)
    this.status = status
  }
}
export async function api<T>(
  path: string,
  options: RequestInit = {}
): Promise<T> {
  let response: Response
  try {
    response = await fetch(path, {
      ...options,
      credentials: "same-origin",
      cache: "no-store",
      headers: {
        ...(options.body ? { "Content-Type": "application/json" } : {}),
        ...options.headers,
      },
    })
  } catch {
    throw new Error(
      "Could not reach your studio. Check your connection and try again."
    )
  }
  const payload = await response.json().catch(() => null)
  if (!response.ok) {
    if (response.status === 401 && path !== "/api/auth")
      window.dispatchEvent(new Event("studio-locked"))
    throw new ApiError(
      typeof payload?.detail === "string"
        ? payload.detail
        : "The request could not be completed. Please try again.",
      response.status
    )
  }
  return payload
}
export const post = <T>(path: string, body: unknown) =>
  api<T>(path, { method: "POST", body: JSON.stringify(body) })
export const downloadUrl = (url: string) =>
  url + (url.includes("?") ? "&" : "?") + "download=1&v=2"
export const active = (job: Job) =>
  job.status === "queued" || job.status === "training"
export const modelName = (job: Job) =>
  job.model_size === "both"
    ? "Full + Lite"
    : job.model_size === "lite"
      ? "Lite"
      : "Full"
export const percent = (job: Job) =>
  Math.round(Math.max(0, Math.min(1, job.progress || 0)) * 100)
export const metric = (value?: number) => window.NAMLossChart.format(value)
export const clock = (seconds: number, decimals = false) => {
  const value = Math.max(0, seconds || 0)
  return (
    Math.floor(value / 60) +
    ":" +
    String(Math.floor(value % 60)).padStart(2, "0") +
    (decimals ? "." + Math.floor((value % 1) * 10) : "")
  )
}
export const date = (value: string, time = false) =>
  new Date(value).toLocaleString([], {
    month: "short",
    day: "numeric",
    ...(time ? { hour: "numeric", minute: "2-digit" } : {}),
  })
export function trainingIssue(take: Capture) {
  if (take.status !== "ready")
    return "This recording was interrupted. Record a fresh pair."
  if (take.clipped)
    return "This audio is clipping. Lower the input level and record again."
  if (take.duration < 12) return "Training needs at least 12 seconds of audio."
  return ""
}
