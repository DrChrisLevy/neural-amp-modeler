import { useState } from "react"
import {
  ArrowLeft,
  AudioLines,
  Info,
  LoaderCircle,
  Mic2,
  Plug2,
  Square,
} from "lucide-react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Waveform } from "@/components/comparison"
import { clock, post } from "@/lib/studio"
import { cn } from "@/lib/utils"
import type { Studio } from "@/hooks/use-studio"
import type { Capture } from "@/lib/types"

export function Recording({
  studio,
  initialName,
  onBack,
  onConnection,
}: {
  studio: Studio
  initialName: string
  onBack: () => void
  onConnection: () => void
}) {
  const [name, setName] = useState(initialName)
  const [busy, setBusy] = useState("")
  const [error, setError] = useState("")
  const bridge = studio.state.bridge
  async function record() {
    if (!bridge.connected && !bridge.recording) {
      onConnection()
      return
    }
    const action = bridge.recording ? "stop" : "start"
    setBusy(action)
    setError("")
    try {
      await post<{ capture?: Capture }>("/api/captures/record", {
        action,
        name: name.trim() || "Untitled take",
      })
      studio.setBridge({
        recording: action === "start",
        ...(action === "start" ? { samples: 0 } : {}),
      })
      await studio.refresh()
      if (action === "stop") onBack()
    } catch (error) {
      setError((error as Error).message)
      void studio.refresh()
    } finally {
      setBusy("")
    }
  }
  return (
    <section className="page-enter mx-auto max-w-3xl space-y-7">
      <Button
        variant="ghost"
        size="sm"
        className="-ml-2 text-muted-foreground"
        onClick={onBack}
      >
        <ArrowLeft className="size-3.5" />
        All pairs
      </Button>
      <div>
        <h1 className="text-3xl font-medium tracking-tight">Record a pair</h1>
        <p className="mt-3 text-sm text-muted-foreground">
          Record your guitar input and processed output together.
        </p>
      </div>
      <div className="overflow-hidden rounded-xl border bg-card">
        <div className="flex items-center justify-between border-b px-6 py-4 text-xs">
          <span className="flex items-center gap-2">
            <span
              className={cn(
                "size-1.5 rounded-full",
                bridge.connected ? "bg-emerald-400" : "bg-muted-foreground"
              )}
            />
            {bridge.device || "No audio interface connected"}
          </span>
          <Button
            size="icon-sm"
            variant="ghost"
            onClick={onConnection}
            aria-label="Audio connection help"
          >
            <Info className="size-4" />
          </Button>
        </div>
        <div className="space-y-7 p-6 sm:p-8">
          <div className="space-y-2">
            <Label htmlFor="record-name">Pair name</Label>
            <Input
              id="record-name"
              value={name}
              onChange={(event) => setName(event.target.value)}
              disabled={bridge.recording || !!busy}
              placeholder="Name this pair"
              maxLength={100}
              className="h-10"
            />
          </div>
          <div className="py-4 text-center">
            <div className="mb-4 flex items-center justify-center gap-2 text-xs text-muted-foreground">
              {bridge.recording && (
                <span className="size-2 animate-pulse rounded-full bg-primary" />
              )}
              {busy === "stop"
                ? "Saving your recording…"
                : bridge.recording
                  ? "Recording"
                  : "Input monitor"}
            </div>
            <div className="font-mono text-6xl font-light tracking-tighter tabular-nums">
              {clock(
                bridge.recording || busy === "stop"
                  ? bridge.samples / bridge.sample_rate
                  : 0
              )}
            </div>
          </div>
          <div className="h-25">
            <Waveform
              peaks={studio.peaks}
              scale={1}
              active={bridge.connected}
            />
          </div>
          <div className="grid grid-cols-2 gap-5">
            {(["dry", "wet"] as const).map((channel) => (
              <div key={channel}>
                <div className="mb-2 flex justify-between text-xs text-muted-foreground">
                  <span>
                    {channel === "dry" ? "Dry input" : "Processed output"}
                  </span>
                  <AudioLines className="size-3" />
                </div>
                <div className="h-1 overflow-hidden rounded-full bg-muted">
                  <div
                    className="h-full bg-primary transition-[width] duration-100"
                    style={{
                      width:
                        Math.min(
                          100,
                          Math.sqrt(bridge.levels?.[channel] || 0) * 100
                        ) + "%",
                    }}
                  />
                </div>
              </div>
            ))}
          </div>
          {error && (
            <p role="alert" className="text-sm text-destructive">
              {error}
            </p>
          )}
          <div className="flex justify-center border-t pt-6">
            <Button
              size="lg"
              disabled={!!busy}
              onClick={record}
              className="h-11 min-w-44"
            >
              {busy ? (
                <LoaderCircle className="animate-spin" />
              ) : bridge.recording ? (
                <Square className="size-4 fill-current" />
              ) : bridge.connected ? (
                <Mic2 className="size-4" />
              ) : (
                <Plug2 className="size-4" />
              )}
              {busy === "stop"
                ? "Saving…"
                : busy
                  ? "Starting…"
                  : bridge.recording
                    ? "Stop recording"
                    : bridge.connected
                      ? "Start recording"
                      : "Connect interface"}
            </Button>
          </div>
        </div>
      </div>
      <p className="flex items-start gap-2 text-xs leading-relaxed text-muted-foreground">
        <Info className="mt-0.5 size-3.5 shrink-0" />
        Play at a range of levels for at least 12 seconds. Keep the preset fixed
        and turn off delay and reverb.
      </p>
    </section>
  )
}
