export type Capture = {
  id: string
  name: string
  created_at: string
  duration: number
  sample_rate: number
  source: string
  status: string
  clipped: boolean
  filenames?: { dry: string; wet: string }
  peaks?: { dry: number[]; wet: number[] }
  dry_url: string
  wet_url: string
}
export type Point = {
  epoch: number
  loss: number
  train_loss?: number
  partial?: boolean
  step?: number
}
export type Job = {
  id: string
  capture_id: string
  status: "queued" | "training" | "completed" | "failed"
  stage?: string
  name?: string
  model_size: "full" | "lite" | "both"
  epochs: number
  epoch?: number
  progress?: number
  created_at: string
  elapsed_seconds?: number
  best_loss?: number
  best_epoch?: number
  error?: string
  metrics?: Point[]
  training_metrics?: Point[]
  train_epoch_loss?: number
  model_url?: string
  prediction_url?: string
  reference_url?: string
  dry_url?: string
}
export type Bridge = {
  connected: boolean
  recording: boolean
  device?: string
  capture_id?: string
  samples: number
  sample_rate: number
  levels?: { dry: number; wet: number }
}
export type StudioState = {
  captures: Capture[]
  jobs: Job[]
  bridge: Bridge
  revision?: string
}
export type Config = {
  authenticated: boolean
  auth_required: boolean
  training_available: boolean
}
export type Upload = {
  id: string
  name: string
  dry: File
  wet: File
  status: "pending" | "uploading" | "processing" | "failed"
  progress: number
  error: string
}
export type Source = "dry" | "reference" | "model"
export type Track = {
  state: "loading" | "ready" | "error" | "unavailable"
  buffer?: AudioBuffer
  peaks?: number[]
  error?: string
}
export interface AudioEngine {
  key: string | null
  tracks: Partial<Record<Source, Track>>
  active: Source
  currentTime: number
  duration: number
  playing: boolean
  loading: boolean
  context?: AudioContext
  load: (key: string, urls: Partial<Record<Source, string>>) => Promise<void>
  play: (source?: Source, position?: number | null) => Promise<void>
  pause: () => void
  reset: () => void
}
export interface UploadManager {
  items: Map<string, Upload>
  active: boolean
  add: (name: string, dry: File, wet: File) => string
  start: (id: string) => void
}
declare global {
  interface Window {
    NAMUploads: new (options: {
      changed: () => void
      completed: (capture: Capture) => void
      unauthorized: () => void
    }) => UploadManager
    NAMComparisonAudio: new (options: {
      changed: () => void
      unauthorized: () => void
    }) => AudioEngine
    NAMLossChart: {
      format: (value?: number) => string
      trainingSeries: (job: Job) => { points: Point[]; averaged: boolean }
    }
  }
}
