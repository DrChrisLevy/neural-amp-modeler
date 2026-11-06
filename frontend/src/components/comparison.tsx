import {
  useEffect,
  useRef,
  useState,
  type KeyboardEvent,
  type PointerEvent,
} from "react"
import {
  AudioLines,
  Headphones,
  LoaderCircle,
  Pause,
  Play,
  RotateCcw,
  Volume2,
} from "lucide-react"
import { Button } from "@/components/ui/button"
import { useTheme } from "@/components/theme-provider"
import { clock } from "@/lib/studio"
import { cn } from "@/lib/utils"
import type { AudioEngine, Capture, Job, Source, Track } from "@/lib/types"

const sources: Source[] = ["dry", "reference", "model"]
const labels = { dry: "Dry input", reference: "Reference", model: "NAM model" }
const descriptions = {
  dry: "Before your gear",
  reference: "Recorded output",
  model: "Learned output",
}
const empty: Partial<Record<Source, Track>> = {}

export function Waveform({
  peaks = [],
  scale = 1,
  active = true,
}: {
  peaks?: number[]
  scale?: number
  active?: boolean
}) {
  const canvas = useRef<HTMLCanvasElement>(null)
  const { theme } = useTheme()
  useEffect(() => {
    const element = canvas.current
    if (!element) return
    const draw = () => {
      const { width, height } = element.getBoundingClientRect()
      const ratio = window.devicePixelRatio || 1
      element.width = width * ratio
      element.height = height * ratio
      const context = element.getContext("2d")
      if (!context) return
      context.scale(ratio, ratio)
      context.fillStyle = active
        ? theme === "dark"
          ? "#ed987f"
          : "#ba5039"
        : theme === "dark"
          ? "#626c64"
          : "#a1aaa2"
      const bars = Math.max(1, Math.floor(width / 4))
      for (let i = 0; i < bars; i++) {
        const start = Math.floor((i * peaks.length) / bars)
        const end = Math.max(
          start + 1,
          Math.floor(((i + 1) * peaks.length) / bars)
        )
        const peak = Math.max(0, ...peaks.slice(start, end))
        const size = Math.max(2, Math.min(1, peak / scale) * (height - 16))
        context.fillRect((i * width) / bars, (height - size) / 2, 2, size)
      }
    }
    const observer = new ResizeObserver(draw)
    observer.observe(element)
    draw()
    return () => observer.disconnect()
  }, [peaks, scale, active, theme])
  return (
    <canvas ref={canvas} aria-hidden="true" className="block h-full w-full" />
  )
}

export function Comparison({ take, job }: { take: Capture; job?: Job }) {
  const engine = useRef<AudioEngine | null>(null)
  const [snapshot, setSnapshot] = useState({
    playing: false,
    loading: true,
    active: "reference" as Source,
    duration: 0,
    tracks: empty,
  })
  const [position, setPosition] = useState(0)
  const [error, setError] = useState("")
  const [retry, setRetry] = useState(0)
  const completed = job?.status === "completed"
  const dry = completed ? job.dry_url : take.dry_url
  const reference = completed ? job.reference_url : take.wet_url
  const model = completed ? job.prediction_url : undefined
  const key = take.id + ":" + (completed ? job.id : "capture") + ":" + retry
  useEffect(() => {
    let disposed = false
    const audio = new window.NAMComparisonAudio({
      changed: () => {
        if (disposed) return
        setSnapshot({
          playing: audio.playing,
          loading: audio.loading,
          active: audio.active,
          duration: audio.duration,
          tracks: { ...audio.tracks },
        })
        setPosition(audio.currentTime)
      },
      unauthorized: () => window.dispatchEvent(new Event("studio-locked")),
    })
    engine.current = audio
    void audio.load(key, { dry, reference, model })
    return () => {
      disposed = true
      audio.reset()
      void audio.context?.close()
      engine.current = null
    }
  }, [key, dry, reference, model])
  useEffect(() => {
    if (!snapshot.playing) return
    let frame: number
    const tick = () => {
      setPosition(engine.current?.currentTime || 0)
      frame = requestAnimationFrame(tick)
    }
    frame = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(frame)
  }, [snapshot.playing])
  const play = (source = snapshot.active, time?: number) => {
    setError("")
    void engine.current
      ?.play(source, time)
      .catch(() =>
        setError(
          "Playback could not start. Check your audio output and try again."
        )
      )
  }
  const toggle = () => {
    if (engine.current?.playing) engine.current.pause()
    else play()
  }
  useEffect(() => {
    const keyboard = (event: globalThis.KeyboardEvent) => {
      const target = event.target as HTMLElement
      if (
        event.code !== "Space" ||
        target.closest(
          "button,input,select,textarea,[role=slider],[role=dialog],[role=alertdialog]"
        )
      )
        return
      event.preventDefault()
      const audio = engine.current
      if (audio?.playing) audio.pause()
      else
        void audio
          ?.play()
          .catch(() => setError("Playback could not start. Try again."))
    }
    window.addEventListener("keydown", keyboard)
    return () => window.removeEventListener("keydown", keyboard)
  }, [])
  const seek = (event: PointerEvent<HTMLDivElement>, source: Source) => {
    const bounds = event.currentTarget.getBoundingClientRect()
    play(
      source,
      Math.max(0, Math.min(1, (event.clientX - bounds.left) / bounds.width)) *
        snapshot.duration
    )
  }
  const keyboardSeek = (
    event: KeyboardEvent<HTMLDivElement>,
    source: Source
  ) => {
    if (event.code === "Space") {
      event.preventDefault()
      toggle()
      return
    }
    if (
      ["ArrowLeft", "ArrowRight", "Home", "End", "Enter"].includes(event.key)
    ) {
      event.preventDefault()
      const step = event.shiftKey ? 5 : 0.25
      play(
        source,
        event.key === "Home"
          ? 0
          : event.key === "End"
            ? snapshot.duration
            : event.key === "Enter"
              ? position
              : Math.max(
                  0,
                  Math.min(
                    snapshot.duration,
                    position + (event.key === "ArrowLeft" ? -step : step)
                  )
                )
      )
    }
  }
  const scale = Math.max(
    0.01,
    ...sources.flatMap((source) => snapshot.tracks[source]?.peaks || [])
  )
  const fraction = snapshot.duration ? (position / snapshot.duration) * 100 : 0
  const failed = sources.some(
    (source) => snapshot.tracks[source]?.state === "error"
  )
  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h2 className="flex items-center gap-2 text-sm font-medium">
            <Headphones className="size-4 text-muted-foreground" />
            Audio comparison
          </h2>
          <p className="mt-1.5 text-xs text-muted-foreground">
            Click or drag a waveform to listen. Switch rows to compare the same
            moment.
          </p>
        </div>
        <div className="flex items-center gap-3">
          <span className="font-mono text-xs text-muted-foreground tabular-nums">
            {clock(position, true)} <span className="mx-1 opacity-50">/</span>{" "}
            {clock(snapshot.duration, true)}
          </span>
          <Button
            variant="outline"
            size="icon"
            onClick={toggle}
            disabled={snapshot.loading || !snapshot.duration}
            aria-label={snapshot.playing ? "Pause audio" : "Play audio"}
          >
            {snapshot.loading ? (
              <LoaderCircle className="animate-spin" />
            ) : snapshot.playing ? (
              <Pause className="size-4" />
            ) : (
              <Play className="size-4" />
            )}
          </Button>
        </div>
      </div>
      <div className="overflow-hidden rounded-xl border bg-card">
        {sources.map((source, index) => {
          const track = snapshot.tracks[source]
          const ready = track?.state === "ready" && !snapshot.loading
          const selected = snapshot.active === source
          return (
            <div
              key={source}
              className={cn(
                "grid transition-colors sm:grid-cols-[155px_1fr]",
                index !== 0 && "border-t",
                selected && "bg-primary/[.035]"
              )}
            >
              <button
                disabled={!ready}
                onClick={() => play(source)}
                aria-label={"Listen to " + labels[source]}
                aria-pressed={selected}
                className={cn(
                  "flex items-center justify-between gap-3 p-5 text-left transition-colors outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-inset disabled:cursor-default sm:border-r",
                  selected
                    ? "text-primary"
                    : "text-muted-foreground hover:text-foreground"
                )}
              >
                <span>
                  <span className="block text-xs font-medium">
                    {labels[source]}
                  </span>
                  <span className="mt-1.5 block text-[10px] text-muted-foreground">
                    {descriptions[source]}
                  </span>
                </span>
                {selected && snapshot.playing ? (
                  <Volume2 className="size-3.5 shrink-0" />
                ) : (
                  <AudioLines className="size-3.5 shrink-0 opacity-50" />
                )}
              </button>
              <div className="px-5 py-4 sm:px-6">
                <div
                  role="slider"
                  tabIndex={ready ? 0 : -1}
                  aria-label={"Seek " + labels[source]}
                  aria-disabled={!ready}
                  aria-valuemin={0}
                  aria-valuemax={snapshot.duration}
                  aria-valuenow={position}
                  aria-valuetext={clock(position, true)}
                  className={cn(
                    "relative h-24 rounded-sm outline-none focus-visible:ring-2 focus-visible:ring-ring",
                    ready && "cursor-crosshair touch-none"
                  )}
                  onPointerDown={(event) => {
                    if (!ready || event.button !== 0) return
                    event.currentTarget.setPointerCapture(event.pointerId)
                    event.currentTarget.focus()
                    seek(event, source)
                  }}
                  onPointerMove={(event) => {
                    if (
                      ready &&
                      event.currentTarget.hasPointerCapture(event.pointerId)
                    )
                      seek(event, source)
                  }}
                  onPointerUp={(event) => {
                    if (event.currentTarget.hasPointerCapture(event.pointerId))
                      event.currentTarget.releasePointerCapture(event.pointerId)
                  }}
                  onKeyDown={(event) => keyboardSeek(event, source)}
                >
                  {track?.state === "ready" ? (
                    <>
                      <Waveform
                        peaks={track.peaks}
                        scale={scale}
                        active={selected}
                      />
                      <div
                        className="pointer-events-none absolute inset-y-1 left-0 bg-primary/5"
                        style={{ width: fraction + "%" }}
                      />
                      <div
                        className={cn(
                          "pointer-events-none absolute inset-y-0 w-px",
                          selected ? "bg-primary" : "bg-muted-foreground/50"
                        )}
                        style={{ left: fraction + "%" }}
                      >
                        <span
                          className={cn(
                            "absolute -top-0.5 -left-0.5 size-1.5 rounded-full",
                            selected ? "bg-primary" : "bg-muted-foreground"
                          )}
                        />
                      </div>
                    </>
                  ) : (
                    <div className="flex h-full items-center justify-center gap-2 text-xs text-muted-foreground">
                      {track?.state === "loading" || !track ? (
                        <>
                          <LoaderCircle className="size-3.5 animate-spin" />
                          Loading audio…
                        </>
                      ) : track.state === "error" ? (
                        track.error
                      ) : (
                        "Available after training"
                      )}
                    </div>
                  )}
                </div>
              </div>
            </div>
          )
        })}
        <div className="flex items-center justify-between border-t px-5 py-3 text-[11px] text-muted-foreground">
          <span>
            {completed
              ? "Held-out audio · shared playback position"
              : "Original capture · shared playback position"}
          </span>
          <span className="hidden items-center gap-2 sm:flex">
            <kbd className="rounded border bg-muted px-1.5 py-0.5 font-mono text-[10px]">
              space
            </kbd>{" "}
            Play / pause
          </span>
        </div>
      </div>
      {(failed || error) && (
        <div
          role="alert"
          className="flex items-center justify-between gap-3 rounded-lg border border-destructive/20 bg-destructive/5 p-3 text-xs text-destructive"
        >
          <span>
            {error ||
              "Some audio could not load. You can retry without leaving this page."}
          </span>
          <Button
            size="sm"
            variant="outline"
            onClick={() => {
              setRetry((value) => value + 1)
              setError("")
            }}
          >
            <RotateCcw className="size-3.5" />
            Retry
          </Button>
        </div>
      )}
    </div>
  )
}
