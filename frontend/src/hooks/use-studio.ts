import { useCallback, useEffect, useRef, useState } from "react"
import { api } from "@/lib/studio"
import type {
  Bridge,
  Capture,
  Config,
  Job,
  StudioState,
  Upload,
} from "@/lib/types"

const empty: StudioState = {
  captures: [],
  jobs: [],
  bridge: {
    connected: false,
    recording: false,
    samples: 0,
    sample_rate: 48000,
  },
}

export function useStudio() {
  const [config, setConfig] = useState<Config | null>(null)
  const [state, setState] = useState<StudioState>(empty)
  const [loaded, setLoaded] = useState(false)
  const [offline, setOffline] = useState("")
  const [peaks, setPeaks] = useState<number[]>([])
  const [uploads, setUploads] = useState<Upload[]>([])
  const revision = useRef("")
  const refreshing = useRef<Promise<void> | null>(null)

  const refresh = useCallback(() => {
    if (refreshing.current) return refreshing.current
    refreshing.current = api<Partial<StudioState>>(
      "/api/state" +
        (revision.current
          ? "?revision=" + encodeURIComponent(revision.current)
          : "")
    )
      .then((next) => {
        revision.current = next.revision || ""
        setState((previous) => ({
          captures: next.captures || previous.captures,
          jobs: next.jobs || previous.jobs,
          bridge: next.bridge || previous.bridge,
        }))
        setLoaded(true)
        setOffline("")
      })
      .catch((error) => {
        setOffline(error.message)
      })
      .finally(() => {
        refreshing.current = null
      })
    return refreshing.current
  }, [])

  const initialize = useCallback(async () => {
    try {
      setConfig(await api<Config>("/api/config"))
      setOffline("")
    } catch (error) {
      setOffline((error as Error).message)
    }
  }, [])
  useEffect(() => {
    // initialize only sets state after the config request resolves.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    void initialize()
  }, [initialize])

  useEffect(() => {
    if (!config) return
    let stopped = false
    let socket: WebSocket | null = null
    let reconnect: ReturnType<typeof setTimeout>
    let polling: ReturnType<typeof setTimeout>
    let heartbeat: ReturnType<typeof setInterval>
    const poll = async () => {
      if (!document.hidden) await refresh()
      if (!stopped) polling = setTimeout(poll, 2000)
    }
    const connect = () => {
      if (stopped || (socket && socket.readyState < WebSocket.CLOSING)) return
      socket = new WebSocket(
        (location.protocol === "https:" ? "wss://" : "ws://") +
          location.host +
          "/ws/monitor"
      )
      socket.onopen = () => {
        heartbeat = setInterval(() => {
          if (socket?.readyState === WebSocket.OPEN) socket.send("ping")
        }, 25000)
      }
      socket.onmessage = (event) => {
        let message
        try {
          message = JSON.parse(event.data)
        } catch {
          return
        }
        if (message.type === "state" && message.bridge) {
          setState((previous) => ({
            ...previous,
            bridge: { ...previous.bridge, ...message.bridge },
          }))
          if (!message.bridge.connected) setPeaks([])
          void refresh()
        }
        if (message.type === "levels") {
          setState((previous) => ({
            ...previous,
            bridge: {
              ...previous.bridge,
              levels: {
                dry: Number(message.dry || 0),
                wet: Number(message.wet || 0),
              },
              ...(Number.isFinite(message.samples)
                ? { samples: message.samples }
                : {}),
            },
          }))
          if (Array.isArray(message.peaks))
            setPeaks((previous) =>
              [...previous, ...message.peaks.filter(Number.isFinite)].slice(
                -240
              )
            )
        }
      }
      socket.onclose = () => {
        clearInterval(heartbeat)
        socket = null
        if (!stopped) reconnect = setTimeout(connect, 3000)
      }
    }
    const visible = () => {
      if (!document.hidden) {
        void refresh()
        connect()
      }
    }
    void poll()
    connect()
    document.addEventListener("visibilitychange", visible)
    return () => {
      stopped = true
      clearTimeout(polling)
      clearTimeout(reconnect)
      clearInterval(heartbeat)
      socket?.close()
      document.removeEventListener("visibilitychange", visible)
    }
  }, [config, refresh])

  // The manager stores callbacks; it does not invoke them during construction.
  // eslint-disable-next-line react-hooks/refs
  const [uploader] = useState(() => {
    const manager = new window.NAMUploads({
      changed: () =>
        setUploads([...manager.items.values()].map((item) => ({ ...item }))),
      completed: (take: Capture) => {
        setState((previous) => ({
          ...previous,
          captures: [
            take,
            ...previous.captures.filter((item) => item.id !== take.id),
          ],
        }))
        void refresh()
      },
    })
    return manager
  })
  const uploading = uploads.some(
    (item) => item.status === "uploading" || item.status === "processing"
  )
  useEffect(() => {
    const prevent = (event: BeforeUnloadEvent) => {
      if (state.bridge.recording || uploading) {
        event.preventDefault()
        event.returnValue = ""
      }
    }
    window.addEventListener("beforeunload", prevent)
    return () => window.removeEventListener("beforeunload", prevent)
  }, [state.bridge.recording, uploading])

  const addJob = (job: Job) =>
    setState((previous) => ({
      ...previous,
      jobs: [job, ...previous.jobs.filter((item) => item.id !== job.id)],
    }))
  const removePair = (id: string) =>
    setState((previous) => ({
      ...previous,
      captures: previous.captures.filter((item) => item.id !== id),
      jobs: previous.jobs.filter((item) => item.capture_id !== id),
    }))
  const setBridge = (bridge: Partial<Bridge>) =>
    setState((previous) => ({
      ...previous,
      bridge: { ...previous.bridge, ...bridge },
    }))
  return {
    config,
    state,
    loaded,
    offline,
    peaks,
    uploads,
    uploader,
    refresh,
    initialize,
    addJob,
    removePair,
    setBridge,
  }
}

export type Studio = ReturnType<typeof useStudio>
