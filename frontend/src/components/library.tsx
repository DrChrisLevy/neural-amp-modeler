import { Fragment, useMemo, useState } from "react"
import {
  ArrowDownToLine,
  ArrowUpRight,
  AudioLines,
  Check,
  ChevronDown,
  ChevronRight,
  Clock3,
  LoaderCircle,
  Plus,
  Search,
  SlidersHorizontal,
  Trash2,
} from "lucide-react"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Input } from "@/components/ui/input"
import { Progress } from "@/components/ui/progress"
import { Skeleton } from "@/components/ui/skeleton"
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table"
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuRadioGroup,
  DropdownMenuRadioItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip"
import {
  active,
  clock,
  date,
  downloadUrl,
  metric,
  modelName,
  percent,
} from "@/lib/studio"
import type { Capture, Job, Upload } from "@/lib/types"
import type { Studio } from "@/hooks/use-studio"
import { cn } from "@/lib/utils"

export function StatusBadge({ job }: { job: Job }) {
  const running = active(job)
  return (
    <Badge
      variant="outline"
      className={cn(
        "gap-1.5 font-normal",
        running
          ? "border-primary/20 bg-primary/8 text-primary"
          : job.status === "completed"
            ? "border-emerald-500/20 bg-emerald-500/7 text-emerald-600 dark:text-emerald-300"
            : "border-destructive/20 text-destructive"
      )}
    >
      {running ? (
        <LoaderCircle className="size-3 animate-spin" />
      ) : job.status === "completed" ? (
        <Check className="size-3" />
      ) : null}
      {job.status === "training"
        ? "Training " + percent(job) + "%"
        : job.status === "completed"
          ? "Trained"
          : job.status === "queued"
            ? "Queued"
            : "Failed"}
    </Badge>
  )
}

function PairMark({ take }: { take: Capture }) {
  const peaks = take.peaks?.wet || []
  const max = Math.max(0.01, ...peaks)
  return (
    <div className="pair-mark flex size-11 shrink-0 items-center justify-center rounded-xl border border-primary/12 bg-primary/6 text-primary">
      {peaks.length ? (
        <svg viewBox="0 0 36 32" className="size-8" aria-hidden="true">
          {Array.from({ length: 9 }, (_, i) => {
            const value =
              Math.max(
                ...peaks.slice(
                  Math.floor((i * peaks.length) / 9),
                  Math.max(1, Math.floor(((i + 1) * peaks.length) / 9))
                )
              ) / max
            return (
              <line
                key={i}
                x1={3 + i * 3.75}
                x2={3 + i * 3.75}
                y1={16 - Math.max(2, value * 11)}
                y2={16 + Math.max(2, value * 11)}
                stroke="currentColor"
                strokeWidth="1.5"
                strokeLinecap="round"
              />
            )
          })}
        </svg>
      ) : (
        <AudioLines className="size-5" />
      )}
    </div>
  )
}

function FileLink({
  take,
  channel,
}: {
  take: Capture
  channel: "dry" | "wet"
}) {
  return (
    <a
      href={downloadUrl(take[channel === "dry" ? "dry_url" : "wet_url"])}
      download
      aria-label={
        "Download " +
        (channel === "dry" ? "input" : "output") +
        " for " +
        take.name
      }
      className="group inline-flex max-w-full items-center gap-2.5 rounded-md py-2 text-muted-foreground transition-colors hover:text-foreground focus-visible:outline-2 focus-visible:outline-ring"
    >
      <span className="rounded border px-1.5 py-1 font-mono text-[9px] tracking-wider text-muted-foreground group-hover:border-primary/50 group-hover:text-primary">
        WAV
      </span>
      <span
        className="max-w-36 truncate text-xs"
        title={take.filenames?.[channel]}
      >
        {take.filenames?.[channel] || channel + ".wav"}
      </span>
    </a>
  )
}

function UploadRow({ item, retry }: { item: Upload; retry: () => void }) {
  const failed = item.status === "failed"
  return (
    <TableRow className="bg-primary/[.025]">
      <TableCell className="py-6 pl-6">
        <div className="flex items-center gap-3.5">
          <div className="flex size-11 items-center justify-center rounded-xl border border-dashed border-primary/30 text-primary">
            <LoaderCircle className={cn("size-5", !failed && "animate-spin")} />
          </div>
          <div>
            <div className="font-medium">{item.name}</div>
            <div
              className={cn(
                "mt-1 max-w-56 text-xs",
                failed ? "text-destructive" : "text-muted-foreground"
              )}
            >
              {failed ? item.error : "Uploading audio pair"}
            </div>
          </div>
        </div>
      </TableCell>
      <TableCell>
        <span className="block max-w-40 truncate text-xs text-muted-foreground">
          {item.dry.name}
        </span>
      </TableCell>
      <TableCell>
        <span className="block max-w-40 truncate text-xs text-muted-foreground">
          {item.wet.name}
        </span>
      </TableCell>
      <TableCell>
        <div className="max-w-44 space-y-2" role="status">
          <span className="text-xs text-primary">
            {failed
              ? "Upload failed"
              : item.status === "processing"
                ? "Processing audio…"
                : "Uploading " + item.progress + "%"}
          </span>
          {!failed && (
            <Progress
              value={item.progress}
              aria-label={"Upload progress for " + item.name}
              className="h-1"
            />
          )}
        </div>
      </TableCell>
      <TableCell>
        {failed && (
          <Button variant="outline" size="sm" onClick={retry}>
            Retry
          </Button>
        )}
      </TableCell>
    </TableRow>
  )
}

type Props = {
  studio: Studio
  onAdd: () => void
  onOpen: (id: string, jobId?: string) => void
  onTrain: (take: Capture) => void
  onDelete: (take: Capture) => void
}
export function Library({ studio, onAdd, onOpen, onTrain, onDelete }: Props) {
  const { state, uploads, loaded } = studio
  const [query, setQuery] = useState("")
  const [sort, setSort] = useState("newest")
  const [expanded, setExpanded] = useState(new Set<string>())
  const pairs = useMemo(
    () =>
      state.captures
        .filter((take) => take.name.toLowerCase().includes(query.toLowerCase()))
        .sort((a, b) =>
          sort === "name"
            ? a.name.localeCompare(b.name)
            : sort === "oldest"
              ? a.created_at.localeCompare(b.created_at)
              : b.created_at.localeCompare(a.created_at)
        ),
    [state.captures, query, sort]
  )
  const pending = uploads.filter(
    (item) => !state.captures.some((take) => take.id === item.id)
  )
  const running = state.jobs.filter(active).length
  const toggle = (id: string) =>
    setExpanded((previous) => {
      const next = new Set(previous)
      if (next.has(id)) next.delete(id)
      else next.add(id)
      return next
    })
  return (
    <section className="page-enter space-y-8" aria-labelledby="library-title">
      <div className="flex flex-wrap items-end justify-between gap-5">
        <h1
          id="library-title"
          className="text-3xl font-medium tracking-tight sm:text-4xl"
        >
          Your Library
        </h1>
        <Button
          size="lg"
          className="h-10 gap-2 px-4 shadow-sm shadow-primary/10"
          onClick={onAdd}
        >
          <Plus className="size-4" /> Add pair
        </Button>
      </div>
      <div className="overflow-hidden rounded-xl border bg-card shadow-sm">
        <div className="flex flex-wrap items-center justify-between gap-4 border-b px-5 py-4 sm:px-6">
          <div className="flex items-center gap-3 text-sm">
            <span className="font-medium">All pairs</span>
            <Badge variant="secondary" className="font-mono text-[11px]">
              {loaded ? state.captures.length : "—"}
            </Badge>
            <span className="hidden text-xs text-muted-foreground sm:inline">
              {state.jobs.length} training{" "}
              {state.jobs.length === 1 ? "run" : "runs"}
            </span>
            {running > 0 && (
              <span className="flex items-center gap-1.5 text-xs text-primary">
                <span className="size-1.5 animate-pulse rounded-full bg-primary" />
                {running} active
              </span>
            )}
          </div>
          <div className="flex items-center gap-2">
            <div className="relative">
              <Search className="absolute top-2.5 left-3 size-3.5 text-muted-foreground" />
              <Input
                aria-label="Find a pair"
                value={query}
                onChange={(event) => setQuery(event.target.value)}
                placeholder="Find a pair…"
                className="h-9 w-44 border-transparent bg-muted/55 pl-9 shadow-none sm:w-56"
              />
            </div>
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button
                  variant="outline"
                  size="icon"
                  aria-label="Sort pairs"
                  className="size-9"
                >
                  <SlidersHorizontal className="size-4" />
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end">
                <DropdownMenuRadioGroup value={sort} onValueChange={setSort}>
                  <DropdownMenuRadioItem value="newest">
                    Newest first
                  </DropdownMenuRadioItem>
                  <DropdownMenuRadioItem value="oldest">
                    Oldest first
                  </DropdownMenuRadioItem>
                  <DropdownMenuRadioItem value="name">
                    Name A–Z
                  </DropdownMenuRadioItem>
                </DropdownMenuRadioGroup>
              </DropdownMenuContent>
            </DropdownMenu>
          </div>
        </div>
        <Table className="min-w-205">
          <TableHeader>
            <TableRow className="bg-muted/25 hover:bg-muted/25">
              <TableHead className="w-[35%] pl-6">Pair</TableHead>
              <TableHead className="w-[17%]">Input</TableHead>
              <TableHead className="w-[17%]">Output</TableHead>
              <TableHead className="w-[22%]">Training</TableHead>
              <TableHead className="pr-6 text-right">
                <span className="sr-only">Actions</span>
              </TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {!loaded &&
              Array.from({ length: 3 }, (_, i) => (
                <TableRow key={i}>
                  <TableCell className="py-7 pl-6">
                    <Skeleton className="h-8 w-48" />
                  </TableCell>
                  {[1, 2, 3, 4].map((j) => (
                    <TableCell key={j}>
                      <Skeleton className="h-5 w-20" />
                    </TableCell>
                  ))}
                </TableRow>
              ))}
            {pending
              .filter((item) =>
                item.name.toLowerCase().includes(query.toLowerCase())
              )
              .map((item) => (
                <UploadRow
                  key={item.id}
                  item={item}
                  retry={() => studio.uploader.start(item.id)}
                />
              ))}
            {pairs.map((take) => {
              const runs = state.jobs.filter(
                (run) => run.capture_id === take.id
              )
              const current = runs.find(active) || runs[0]
              const isOpen = expanded.has(take.id)
              const busy = runs.some(active)
              return (
                <Fragment key={take.id}>
                  <TableRow
                    className={cn(
                      "group transition-colors",
                      isOpen && "bg-muted/20"
                    )}
                  >
                    <TableCell className="py-6 pl-6">
                      <div className="flex items-center gap-3.5">
                        <PairMark take={take} />
                        <div className="min-w-0">
                          <button
                            className="max-w-65 truncate text-left font-medium tracking-[-.01em] transition-colors hover:text-primary focus-visible:outline-2 focus-visible:outline-ring"
                            onClick={() => onOpen(take.id)}
                          >
                            {take.name}
                          </button>
                          <div className="mt-1 flex flex-wrap items-center gap-x-2 text-[11px] text-muted-foreground">
                            <span className="font-mono">
                              {clock(take.duration)}
                            </span>
                            <span>·</span>
                            <span>{date(take.created_at)}</span>
                            <span>·</span>
                            <span>
                              {take.source === "upload"
                                ? "Uploaded"
                                : "Recorded"}
                            </span>
                            {take.status !== "ready" ? (
                              <span className="text-amber-500">
                                Interrupted
                              </span>
                            ) : take.clipped ? (
                              <span className="text-amber-500">Clipping</span>
                            ) : null}
                          </div>
                        </div>
                      </div>
                    </TableCell>
                    <TableCell>
                      <FileLink take={take} channel="dry" />
                    </TableCell>
                    <TableCell>
                      <FileLink take={take} channel="wet" />
                    </TableCell>
                    <TableCell>
                      {current ? (
                        <div className="space-y-1.5">
                          <StatusBadge job={current} />
                          <div className="text-[11px] text-muted-foreground">
                            {modelName(current)} ·{" "}
                            {current.epochs.toLocaleString()} epochs
                          </div>
                          <button
                            onClick={() => toggle(take.id)}
                            aria-expanded={isOpen}
                            aria-controls={"runs-" + take.id}
                            aria-label={
                              (isOpen ? "Hide" : "Show") +
                              " training runs for " +
                              take.name
                            }
                            className="flex items-center gap-1.5 rounded py-0.5 text-xs text-muted-foreground transition-colors hover:text-foreground focus-visible:outline-2 focus-visible:outline-ring"
                          >
                            {isOpen ? "Hide" : "View"} {runs.length}{" "}
                            {runs.length === 1 ? "run" : "runs"}
                            <ChevronDown
                              className={cn(
                                "size-3.5 transition-transform",
                                isOpen && "rotate-180"
                              )}
                            />
                          </button>
                        </div>
                      ) : (
                        <span className="text-xs text-muted-foreground">
                          Not trained
                        </span>
                      )}
                    </TableCell>
                    <TableCell className="pr-6">
                      <div className="flex items-center justify-end gap-1">
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => onTrain(take)}
                          aria-label={"Train " + take.name}
                          className="text-primary"
                        >
                          Train <ArrowUpRight className="size-3.5" />
                        </Button>
                        <Tooltip>
                          <TooltipTrigger asChild>
                            <span tabIndex={busy ? 0 : undefined}>
                              <Button
                                variant="ghost"
                                size="icon-sm"
                                disabled={busy}
                                onClick={() => onDelete(take)}
                                aria-label={"Delete " + take.name}
                                className="text-muted-foreground hover:text-destructive"
                              >
                                <Trash2 className="size-3.5" />
                              </Button>
                            </span>
                          </TooltipTrigger>
                          <TooltipContent>
                            {busy ? "Training in progress" : "Delete pair"}
                          </TooltipContent>
                        </Tooltip>
                      </div>
                    </TableCell>
                  </TableRow>
                  {isOpen && (
                    <TableRow className="hover:bg-transparent">
                      <TableCell colSpan={5} className="bg-muted/20 p-0">
                        <div
                          id={"runs-" + take.id}
                          className="border-l-2 border-primary/40 px-6 py-5"
                        >
                          <div className="mb-3 flex items-center gap-2 text-xs text-muted-foreground">
                            <Clock3 className="size-3.5" />
                            Training history
                          </div>
                          <div className="divide-y rounded-lg border bg-background/40">
                            {runs.map((run) => (
                              <div
                                key={run.id}
                                className="grid grid-cols-[1fr_1fr_1fr_auto] items-center gap-4 px-4 py-3.5"
                              >
                                <button
                                  onClick={() => onOpen(take.id, run.id)}
                                  className="text-left hover:text-primary"
                                >
                                  <span className="text-xs font-medium">
                                    {modelName(run)}
                                    <span className="ml-2 text-muted-foreground">
                                      {run.epochs.toLocaleString()} epochs
                                    </span>
                                  </span>
                                  <span className="mt-1 block text-[11px] text-muted-foreground">
                                    {date(run.created_at, true)}
                                  </span>
                                </button>
                                <div>
                                  <StatusBadge job={run} />
                                  {run.error && (
                                    <p className="mt-1 max-w-56 text-xs text-destructive">
                                      {run.error}
                                    </p>
                                  )}
                                </div>
                                <div className="font-mono text-xs">
                                  <span className="mr-2 text-[10px] text-muted-foreground">
                                    ESR
                                  </span>
                                  {metric(run.best_loss)}
                                </div>
                                <div className="flex items-center gap-2">
                                  {run.status === "completed" &&
                                    run.model_url && (
                                      <Button
                                        variant="outline"
                                        size="sm"
                                        asChild
                                      >
                                        <a
                                          href={downloadUrl(run.model_url)}
                                          download
                                          aria-label={
                                            "Download " +
                                            modelName(run) +
                                            " " +
                                            run.epochs +
                                            " epochs for " +
                                            take.name
                                          }
                                        >
                                          <ArrowDownToLine className="size-3.5" />
                                          .nam
                                        </a>
                                      </Button>
                                    )}
                                  <Button
                                    variant="ghost"
                                    size="icon-sm"
                                    aria-label={
                                      "Open " +
                                      run.epochs +
                                      " epoch run for " +
                                      take.name
                                    }
                                    onClick={() => onOpen(take.id, run.id)}
                                  >
                                    <ChevronRight className="size-4" />
                                  </Button>
                                </div>
                              </div>
                            ))}
                          </div>
                        </div>
                      </TableCell>
                    </TableRow>
                  )}
                </Fragment>
              )
            })}
          </TableBody>
        </Table>
        {loaded && !pairs.length && !pending.length && (
          <div className="flex min-h-65 flex-col items-center justify-center gap-3 px-6 py-16">
            <div className="mb-2 rounded-2xl border bg-muted/40 p-4 text-muted-foreground">
              <AudioLines className="size-6" />
            </div>
            <h2 className="text-sm font-medium">
              {query ? "No matching pairs" : "No pairs yet"}
            </h2>
            <p className="text-center text-sm text-muted-foreground">
              {query
                ? "Try a different name."
                : "Add an input and output WAV to get started."}
            </p>
          </div>
        )}
        <div className="flex items-center justify-between border-t bg-muted/15 px-6 py-3 text-[11px] text-muted-foreground">
          <span>
            {pairs.length} {pairs.length === 1 ? "pair" : "pairs"}
            {pending.length ? " · " + pending.length + " uploading" : ""}
          </span>
        </div>
      </div>
    </section>
  )
}
