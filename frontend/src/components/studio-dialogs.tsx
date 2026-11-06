import { useRef, useState, type FormEvent } from "react"
import {
  AudioLines,
  Check,
  CheckCheck,
  Copy,
  Cpu,
  FileAudio2,
  Info,
  LoaderCircle,
  Mic2,
  Terminal,
  Upload,
} from "lucide-react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Badge } from "@/components/ui/badge"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
import {
  AlertDialog,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { api, post, trainingIssue } from "@/lib/studio"
import type { Capture, Job } from "@/lib/types"
import type { Studio } from "@/hooks/use-studio"
import { cn } from "@/lib/utils"

function ErrorMessage({ message }: { message: string }) {
  return message ? (
    <p
      role="alert"
      className="rounded-lg border border-destructive/20 bg-destructive/5 px-3 py-2.5 text-sm text-destructive"
    >
      {message}
    </p>
  ) : null
}

function FilePicker({
  label,
  description,
  value,
  onChange,
}: {
  label: string
  description: string
  value: File | null
  onChange: (file: File | null) => void
}) {
  const input = useRef<HTMLInputElement>(null)
  const [dragging, setDragging] = useState(false)
  return (
    <div
      onDragOver={(event) => {
        event.preventDefault()
        setDragging(true)
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={(event) => {
        event.preventDefault()
        setDragging(false)
        onChange(event.dataTransfer.files[0] || null)
      }}
      className={cn(
        "relative rounded-xl border border-dashed bg-muted/20 transition-colors",
        dragging && "border-primary bg-primary/5",
        value && "border-solid border-primary/30 bg-primary/[.035]"
      )}
    >
      <input
        ref={input}
        aria-label={label}
        type="file"
        accept=".wav,audio/wav,audio/x-wav"
        className="sr-only"
        tabIndex={-1}
        onChange={(event) => onChange(event.target.files?.[0] || null)}
      />
      <button
        type="button"
        onClick={() => input.current?.click()}
        className="flex w-full flex-col items-center gap-2 rounded-xl p-6 text-center outline-none focus-visible:ring-2 focus-visible:ring-ring"
      >
        <span
          className={cn(
            "mb-2 rounded-lg border bg-background p-2.5",
            value ? "text-primary" : "text-muted-foreground"
          )}
        >
          {value ? (
            <FileAudio2 className="size-5" />
          ) : (
            <Upload className="size-5" />
          )}
        </span>
        <span className="text-sm font-medium">{label}</span>
        <span
          className="w-full truncate text-xs text-muted-foreground"
          title={value?.name}
        >
          {value?.name || description}
        </span>
        <span className="mt-1 text-[11px] text-muted-foreground">
          {value
            ? (value.size / 1024 / 1024).toFixed(1) + " MB · Click to replace"
            : "Choose WAV or drop here"}
        </span>
      </button>
      {value && (
        <Check className="absolute top-3 right-3 size-3.5 text-primary" />
      )}
    </div>
  )
}

export function UploadDialog({
  studio,
  onClose,
  onRecord,
  onAdded,
}: {
  studio: Studio
  onClose: () => void
  onRecord: (name: string) => void
  onAdded: () => void
}) {
  const [name, setName] = useState("")
  const [dry, setDry] = useState<File | null>(null)
  const [wet, setWet] = useState<File | null>(null)
  const [error, setError] = useState("")
  function submit(event: FormEvent) {
    event.preventDefault()
    if (!dry || !wet) {
      setError("Choose both the input and output WAV files.")
      return
    }
    if (![dry, wet].every((file) => file.name.toLowerCase().endsWith(".wav"))) {
      setError("Choose WAV files for both input and output.")
      return
    }
    if ([dry, wet].some((file) => file.size > 128 * 1024 * 1024)) {
      setError("Each WAV must be smaller than 128 MB.")
      return
    }
    studio.uploader.add(name.trim() || "Untitled take", dry, wet)
    onAdded()
    onClose()
  }
  return (
    <Dialog open onOpenChange={(open) => !open && onClose()}>
      <DialogContent className="sm:max-w-xl">
        <DialogHeader>
          <div className="mb-3 flex size-10 items-center justify-center rounded-xl border bg-muted/40">
            <AudioLines className="size-5 text-primary" />
          </div>
          <DialogTitle>Add a pair</DialogTitle>
          <DialogDescription>
            The same signal, before and after your gear.
          </DialogDescription>
        </DialogHeader>
        <form onSubmit={submit} className="space-y-5">
          <div className="space-y-2">
            <Label htmlFor="pair-name">Pair name</Label>
            <Input
              id="pair-name"
              value={name}
              onChange={(event) => setName(event.target.value)}
              placeholder="e.g. Victor’s Mega Squid"
              maxLength={100}
              autoFocus
              className="h-10"
            />
          </div>
          <div className="grid gap-3 sm:grid-cols-2">
            <FilePicker
              label="Input · dry WAV"
              description="Unprocessed signal"
              value={dry}
              onChange={setDry}
            />
            <FilePicker
              label="Output · processed WAV"
              description="Recorded output from your gear"
              value={wet}
              onChange={setWet}
            />
          </div>
          <p className="flex items-start gap-2 text-xs leading-relaxed text-muted-foreground">
            <Info className="mt-0.5 size-3.5 shrink-0" />
            Mono, 48 kHz, matching lengths. Up to 10 minutes and 128 MB per
            file.
          </p>
          <ErrorMessage message={error} />
          <DialogFooter>
            <Button type="button" variant="ghost" onClick={onClose}>
              Cancel
            </Button>
            <Button type="submit" className="h-9 gap-2 px-4">
              <Upload className="size-4" />
              Add pair
            </Button>
          </DialogFooter>
        </form>
        <div className="flex items-center justify-between border-t pt-4">
          <span className="text-xs text-muted-foreground">
            Or capture a live performance
          </span>
          <Button
            variant="ghost"
            size="sm"
            onClick={() => onRecord(name.trim())}
          >
            <Mic2 className="size-3.5" />
            Record a pair
          </Button>
        </div>
      </DialogContent>
    </Dialog>
  )
}

export function TrainDialog({
  take,
  studio,
  onClose,
  onStarted,
}: {
  take: Capture
  studio: Studio
  onClose: () => void
  onStarted: (job: Job) => void
}) {
  const [epochs, setEpochs] = useState("400")
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState("")
  const issue =
    trainingIssue(take) ||
    (!studio.config?.training_available
      ? "Training is available in your deployed Modal studio."
      : "")
  async function submit(event: FormEvent) {
    event.preventDefault()
    if (busy || issue) return
    setBusy(true)
    setError("")
    try {
      const job = await post<Job>("/api/train", {
        capture_id: take.id,
        epochs: Number(epochs),
        model_size: "full",
      })
      studio.addJob(job)
      onStarted(job)
      onClose()
      void studio.refresh()
    } catch (error) {
      setError((error as Error).message)
      setBusy(false)
    }
  }
  return (
    <Dialog open onOpenChange={(open) => !open && !busy && onClose()}>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <div className="mb-3 flex size-10 items-center justify-center rounded-xl border bg-primary/5 text-primary">
            <Cpu className="size-5" />
          </div>
          <DialogTitle>Train a model</DialogTitle>
          <DialogDescription>{take.name}</DialogDescription>
        </DialogHeader>
        <form onSubmit={submit} className="space-y-6">
          <div className="grid grid-cols-2 gap-4">
            <div className="space-y-2">
              <Label>Architecture</Label>
              <div className="flex h-10 items-center justify-between rounded-lg border bg-muted/35 px-3 text-sm">
                A2 Full
                <Badge variant="outline" className="text-[10px]">
                  NAM
                </Badge>
              </div>
            </div>
            <div className="space-y-2">
              <Label htmlFor="training-epochs">Epochs</Label>
              <Select value={epochs} onValueChange={setEpochs}>
                <SelectTrigger id="training-epochs" className="h-10 w-full">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {[50, 100, 400, 500, 1000].map((count) => (
                    <SelectItem value={String(count)} key={count}>
                      {count.toLocaleString()}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          </div>
          <ErrorMessage message={error || issue} />
          <DialogFooter>
            <Button
              type="button"
              variant="ghost"
              onClick={onClose}
              disabled={busy}
            >
              Cancel
            </Button>
            <Button
              type="submit"
              disabled={busy || !!issue}
              className="h-9 px-4"
            >
              {busy ? <LoaderCircle className="animate-spin" /> : <Cpu />}Start
              training
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  )
}

export function DeleteDialog({
  take,
  studio,
  onClose,
}: {
  take: Capture
  studio: Studio
  onClose: () => void
}) {
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState("")
  async function remove() {
    setBusy(true)
    setError("")
    try {
      await api("/api/captures/" + take.id, { method: "DELETE" })
      studio.removePair(take.id)
      onClose()
    } catch (error) {
      setError((error as Error).message)
      setBusy(false)
    }
  }
  return (
    <AlertDialog open onOpenChange={(open) => !open && !busy && onClose()}>
      <AlertDialogContent>
        <AlertDialogHeader>
          <AlertDialogTitle>Delete this pair?</AlertDialogTitle>
          <AlertDialogDescription>
            “{take.name}” and its training runs, models, and audio will be
            permanently deleted.
          </AlertDialogDescription>
        </AlertDialogHeader>
        <ErrorMessage message={error} />
        <AlertDialogFooter>
          <Button variant="outline" disabled={busy} onClick={onClose}>
            Cancel
          </Button>
          <Button variant="destructive" disabled={busy} onClick={remove}>
            {busy && <LoaderCircle className="animate-spin" />}Delete pair
          </Button>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  )
}

export function AuthDialog({ onUnlock }: { onUnlock: () => Promise<void> }) {
  const [key, setKey] = useState("")
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState("")
  async function submit(event: FormEvent) {
    event.preventDefault()
    setBusy(true)
    setError("")
    try {
      await post("/api/auth", { key })
      setKey("")
      await onUnlock()
    } catch (error) {
      setError((error as Error).message)
    } finally {
      setBusy(false)
    }
  }
  return (
    <Dialog open>
      <DialogContent
        showCloseButton={false}
        onEscapeKeyDown={(event) => event.preventDefault()}
        onPointerDownOutside={(event) => event.preventDefault()}
        className="sm:max-w-sm"
      >
        <DialogHeader>
          <DialogTitle>Workspace access</DialogTitle>
          <DialogDescription>
            Enter your workspace key to open the studio.
          </DialogDescription>
        </DialogHeader>
        <form className="space-y-4" onSubmit={submit}>
          <Label htmlFor="workspace-key">Workspace key</Label>
          <Input
            id="workspace-key"
            type="password"
            autoComplete="current-password"
            value={key}
            onChange={(event) => setKey(event.target.value)}
            required
            autoFocus
          />
          <ErrorMessage message={error} />
          <Button type="submit" disabled={busy} className="w-full">
            {busy && <LoaderCircle className="animate-spin" />}Open studio
          </Button>
        </form>
      </DialogContent>
    </Dialog>
  )
}

export function ConnectionDialog({
  studio,
  onClose,
}: {
  studio: Studio
  onClose: () => void
}) {
  const [copied, setCopied] = useState(false)
  const [error, setError] = useState("")
  const connected = studio.state.bridge.connected
  const command =
    "uv run" +
    (studio.config?.auth_required ? " --env-file .env" : "") +
    " neural-amp-modeler bridge --url " +
    location.origin
  async function copy() {
    try {
      await navigator.clipboard.writeText(command)
      setCopied(true)
    } catch {
      setError("Select and copy the command below.")
    }
  }
  return (
    <Dialog open onOpenChange={(open) => !open && onClose()}>
      <DialogContent className="sm:max-w-lg">
        <DialogHeader>
          <div className="mb-3 flex size-10 items-center justify-center rounded-xl border bg-muted/30">
            <Terminal className="size-5" />
          </div>
          <DialogTitle>
            {connected
              ? "Your interface is connected"
              : "Connect your audio interface"}
          </DialogTitle>
          <DialogDescription>
            {connected
              ? studio.state.bridge.device || "Audio interface ready."
              : "Connect the Mini by USB, then run this from your project folder."}
          </DialogDescription>
        </DialogHeader>
        {!connected && (
          <>
            <div className="relative rounded-xl border bg-background p-4 pr-12">
              <code className="font-mono text-xs leading-6 break-words">
                {command}
              </code>
              <Button
                variant="ghost"
                size="icon-sm"
                onClick={copy}
                aria-label="Copy bridge command"
                className="absolute top-3 right-2"
              >
                {copied ? <CheckCheck /> : <Copy />}
              </Button>
            </div>
            <p className="text-xs leading-relaxed text-muted-foreground">
              Keep Terminal open while recording. The connection appears
              automatically. Dry USB 1 · processed USB 3.
            </p>
          </>
        )}
        <div className="flex items-center gap-2.5 rounded-lg border bg-muted/30 p-3 text-sm">
          <span
            className={cn(
              "size-2 rounded-full",
              connected ? "bg-emerald-400" : "animate-pulse bg-amber-400"
            )}
          />
          {connected ? "Ready to record" : "Waiting for the bridge…"}
        </div>
        <ErrorMessage message={error} />
        <DialogFooter>
          <Button variant={connected ? "default" : "outline"} onClick={onClose}>
            {connected ? "Continue" : "Close"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
