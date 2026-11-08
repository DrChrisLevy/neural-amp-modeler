import { useState } from "react"
import {
  Activity,
  ArrowDownToLine,
  ArrowLeft,
  AudioLines,
  ChartNoAxesCombined,
  Clock3,
  Cpu,
  Plus,
  TrendingDown,
} from "lucide-react"
import { Area, AreaChart, CartesianGrid, XAxis, YAxis } from "recharts"
import { Button } from "@/components/ui/button"
import { Card, CardContent } from "@/components/ui/card"
import { Progress } from "@/components/ui/progress"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import {
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
} from "@/components/ui/chart"
import { StatusBadge } from "@/components/library"
import { Comparison } from "@/components/comparison"
import {
  active,
  clock,
  date,
  downloadUrl,
  metric,
  percent,
} from "@/lib/studio"
import type { Capture, Job, Point } from "@/lib/types"

function LossChart({
  title,
  description,
  points,
  epochs,
  logarithmic,
  validation = false,
}: {
  title: string
  description: string
  points: Point[]
  epochs: number
  logarithmic: boolean
  validation?: boolean
}) {
  const valid = points.filter(
    (point) => Number.isFinite(point.loss) && point.loss >= 0
  )
  const positive = valid
    .filter((point) => point.loss > 0)
    .map((point) => point.loss)
  const floor = Math.min(...positive, 1e-6)
  const data = valid.map((point) => ({
    ...point,
    value: logarithmic ? Math.max(floor, point.loss) : point.loss,
  }))
  const color = validation ? "var(--chart-2)" : "var(--chart-1)"
  const id = validation ? "validation-fill" : "training-fill"
  return (
    <Card className="gap-0 overflow-hidden py-0 shadow-none">
      <CardContent className="p-5 sm:p-6">
        <div className="flex items-start justify-between gap-4">
          <div>
            <h3 className="flex items-center gap-2 text-sm font-medium">
              <span
                className="h-0.5 w-3 rounded"
                style={{ background: color }}
              />
              {title}
            </h3>
            <p className="mt-2 text-[11px] text-muted-foreground">
              {description}
            </p>
          </div>
          <span className="font-mono text-xl tracking-tight" style={{ color }}>
            {metric(valid.at(-1)?.loss)}
          </span>
        </div>
        {data.length ? (
          <ChartContainer
            config={{ value: { label: title, color } }}
            className="mt-7 aspect-auto h-60 w-full"
          >
            <AreaChart
              accessibilityLayer
              data={data}
              margin={{ top: 8, right: 8, bottom: 0, left: -14 }}
            >
              <defs>
                <linearGradient id={id} x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor={color} stopOpacity={0.12} />
                  <stop offset="100%" stopColor={color} stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid
                vertical={false}
                strokeDasharray="3 5"
                stroke="var(--border)"
              />
              <XAxis
                dataKey="epoch"
                type="number"
                domain={[0, epochs]}
                tickLine={false}
                axisLine={false}
                tickMargin={12}
                minTickGap={35}
                tick={{ fontSize: 10, fontFamily: "var(--font-mono)" }}
              />
              <YAxis
                scale={logarithmic ? "log" : "linear"}
                domain={logarithmic ? ["auto", "auto"] : [0, "auto"]}
                allowDataOverflow
                tickLine={false}
                axisLine={false}
                tickMargin={8}
                tickCount={4}
                tickFormatter={(value) => metric(value)}
                tick={{ fontSize: 9, fontFamily: "var(--font-mono)" }}
              />
              <ChartTooltip
                content={
                  <ChartTooltipContent
                    labelFormatter={(_, payload) =>
                      "Epoch " +
                      Number(payload[0]?.payload?.epoch || 0).toFixed(0)
                    }
                    formatter={(_, __, item) => (
                      <span className="flex w-full items-center justify-between gap-5">
                        <span className="text-muted-foreground">{title}</span>
                        <span className="font-mono">
                          {metric(item.payload.loss)}
                        </span>
                      </span>
                    )}
                  />
                }
              />
              <Area
                dataKey="value"
                type="linear"
                stroke={color}
                strokeWidth={1.6}
                fill={"url(#" + id + ")"}
                isAnimationActive={false}
                activeDot={{ r: 3, strokeWidth: 0 }}
              />
            </AreaChart>
          </ChartContainer>
        ) : (
          <div className="mt-7 flex h-60 items-center justify-center rounded-lg border border-dashed text-xs text-muted-foreground">
            Measurements appear as training progresses.
          </div>
        )}
        <div className="mt-3 text-right font-mono text-[10px] text-muted-foreground">
          EPOCH
        </div>
      </CardContent>
    </Card>
  )
}

function Training({ job, onTrain }: { job?: Job; onTrain: () => void }) {
  const [logarithmic, setLogarithmic] = useState(true)
  if (!job)
    return (
      <div className="flex min-h-80 flex-col items-center justify-center rounded-xl border border-dashed bg-card/40 p-8 text-center">
        <Cpu className="mb-4 size-7 text-muted-foreground" />
        <h2 className="text-base font-medium">No training runs yet</h2>
        <p className="mt-2 mb-5 text-sm text-muted-foreground">
          Train a NAM model from this audio pair.
        </p>
        <Button onClick={onTrain}>
          <Plus className="size-4" />
          New training run
        </Button>
      </div>
    )
  const training = window.NAMLossChart.trainingSeries(job)
  return (
    <div className="space-y-5">
      <div className="grid gap-3 sm:grid-cols-3">
        <Card className="py-0 shadow-none">
          <CardContent className="p-5">
            <div className="flex items-center justify-between text-xs text-muted-foreground">
              <span>Best validation ESR</span>
              <TrendingDown className="size-4" />
            </div>
            <div className="mt-3 flex items-end gap-3">
              <span className="font-mono text-2xl tracking-tight">
                {metric(job.best_loss)}
              </span>
              {job.best_epoch && (
                <span className="pb-1 text-[11px] text-muted-foreground">
                  epoch {job.best_epoch}
                </span>
              )}
            </div>
          </CardContent>
        </Card>
        <Card className="py-0 shadow-none">
          <CardContent className="p-5">
            <div className="flex items-center justify-between text-xs text-muted-foreground">
              <span>Epochs</span>
              <Activity className="size-4" />
            </div>
            <div className="mt-3 font-mono text-2xl tracking-tight">
              {(job.epoch || 0).toLocaleString()}
              <span className="ml-2 text-sm text-muted-foreground">
                / {job.epochs.toLocaleString()}
              </span>
            </div>
          </CardContent>
        </Card>
        <Card className="py-0 shadow-none">
          <CardContent className="p-5">
            <div className="flex items-center justify-between text-xs text-muted-foreground">
              <span>Training time</span>
              <Clock3 className="size-4" />
            </div>
            <div className="mt-3 font-mono text-2xl tracking-tight">
              {job.elapsed_seconds != null ? clock(job.elapsed_seconds) : "—"}
              <span className="ml-3 font-sans text-[11px] text-muted-foreground">
                A2 Full model
              </span>
            </div>
          </CardContent>
        </Card>
      </div>
      {active(job) && (
        <div className="space-y-3 rounded-xl border bg-card p-5">
          <div className="flex justify-between gap-3 text-xs">
            <span>
              {job.status === "queued"
                ? "Waiting for a GPU…"
                : job.stage === "preparing"
                  ? "Preparing audio and model…"
                  : job.stage === "exporting"
                    ? "Exporting your model…"
                    : "Training model"}
            </span>
            <span className="font-mono text-primary">{percent(job)}%</span>
          </div>
          <Progress
            value={percent(job)}
            className="h-1"
            aria-label="Training progress"
          />
        </div>
      )}
      {job.status === "failed" && (
        <p
          role="alert"
          className="rounded-lg border border-destructive/20 bg-destructive/5 p-4 text-sm text-destructive"
        >
          {job.error ||
            "This training run failed. Start a new run to try again."}
        </p>
      )}
      <div className="flex items-center justify-between pt-3">
        <h2 className="text-sm font-medium">Training metrics</h2>
        <Button
          variant="outline"
          size="sm"
          aria-pressed={logarithmic}
          onClick={() => setLogarithmic((value) => !value)}
          className="h-7 text-[11px]"
        >
          {logarithmic ? "Log scale" : "Linear scale"}
        </Button>
      </div>
      <div className="grid gap-4 lg:grid-cols-2">
        <LossChart
          title="Training loss"
          description={
            "Waveform + spectral · " +
            (training.averaged ? "epoch average" : "measured batches")
          }
          points={training.points}
          epochs={job.epochs}
          logarithmic={logarithmic}
        />
        <LossChart
          title="Validation error"
          description="ESR · audio held out from training"
          points={job.metrics || []}
          epochs={job.epochs}
          logarithmic={logarithmic}
          validation
        />
      </div>
      <p className="text-[11px] leading-relaxed text-muted-foreground">
        Lower is better. Training loss and validation ESR use separate scales.
      </p>
    </div>
  )
}

export function PairDetail({
  take,
  jobs,
  selectedJob,
  onSelectJob,
  tab,
  onTab,
  onBack,
  onTrain,
}: {
  take: Capture
  jobs: Job[]
  selectedJob?: Job
  onSelectJob: (id: string) => void
  tab: string
  onTab: (tab: string) => void
  onBack: () => void
  onTrain: () => void
}) {
  return (
    <section className="page-enter space-y-7">
      <Button
        variant="ghost"
        size="sm"
        className="-ml-2 text-muted-foreground"
        onClick={onBack}
      >
        <ArrowLeft className="size-3.5" />
        All pairs
      </Button>
      <div className="flex flex-wrap items-start justify-between gap-5">
        <div>
          <h1 className="text-3xl font-medium tracking-tight">{take.name}</h1>
          <div className="mt-3 flex flex-wrap items-center gap-2.5 text-xs text-muted-foreground">
            <AudioLines className="size-3.5" />
            <span>{clock(take.duration)}</span>
            <span>·</span>
            <span>
              {take.source === "upload" ? "Uploaded pair" : "Live recording"}
            </span>
            <span>·</span>
            <span>{date(take.created_at)}</span>
          </div>
        </div>
        <div className="flex flex-wrap gap-2">
          {selectedJob?.status === "completed" && selectedJob.model_url && (
            <Button variant="outline" className="h-9" asChild>
              <a href={downloadUrl(selectedJob.model_url)} download>
                <ArrowDownToLine className="size-4" />
                Download .nam
              </a>
            </Button>
          )}
          <Button onClick={onTrain} className="h-9">
            <Plus className="size-4" />
            New training run
          </Button>
        </div>
      </div>
      {jobs.length > 0 && (
        <div className="flex flex-wrap items-center gap-3">
          <label htmlFor="run-picker" className="text-xs text-muted-foreground">
            Training run
          </label>
          <Select value={selectedJob?.id || ""} onValueChange={onSelectJob}>
            <SelectTrigger
              id="run-picker"
              className="w-full bg-card sm:w-auto sm:min-w-80"
            >
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {jobs.map((job) => (
                <SelectItem key={job.id} value={job.id}>
                  A2 Full · {job.epochs.toLocaleString()} epochs ·{" "}
                  {date(job.created_at, true)}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          {selectedJob && <StatusBadge job={selectedJob} />}
        </div>
      )}
      <Tabs value={tab} onValueChange={onTab} className="gap-6">
        <div className="border-b pb-3">
          <TabsList className="h-9">
            <TabsTrigger value="training" className="gap-2 px-4">
              <ChartNoAxesCombined className="size-3.5" />
              Training
            </TabsTrigger>
            <TabsTrigger value="compare" className="gap-2 px-4">
              <AudioLines className="size-3.5" />
              Compare audio
            </TabsTrigger>
          </TabsList>
        </div>
        <TabsContent value="training">
          <Training job={selectedJob} onTrain={onTrain} />
        </TabsContent>
        <TabsContent value="compare">
          <Comparison
            key={take.id + (selectedJob?.id || "")}
            take={take}
            job={selectedJob}
          />
        </TabsContent>
      </Tabs>
    </section>
  )
}
