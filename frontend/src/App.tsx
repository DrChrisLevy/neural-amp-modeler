import { lazy, Suspense, useEffect, useState } from "react"
import { AudioLines, ChevronRight, CircleAlert, Moon, Sun } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Library } from "@/components/library"
const PairDetail = lazy(() =>
  import("@/components/pair-detail").then((module) => ({
    default: module.PairDetail,
  }))
)
import { Recording } from "@/components/recording"
import {
  ConnectionDialog,
  DeleteDialog,
  TrainDialog,
  UploadDialog,
} from "@/components/studio-dialogs"
import { useTheme } from "@/components/theme-provider"
import { useStudio } from "@/hooks/use-studio"
import { active } from "@/lib/studio"
import type { Capture } from "@/lib/types"

function readRoute() {
  const [page, pair = "", run = "", panel] = location.hash.slice(1).split("/")
  return {
    view:
      page === "pair" && pair
        ? ("detail" as const)
        : page === "record"
          ? ("record" as const)
          : ("library" as const),
    pair,
    run,
    tab: panel === "compare" ? "compare" : "training",
  }
}

export default function App() {
  const studio = useStudio()
  const { theme, toggle } = useTheme()
  const [view, setView] = useState<"library" | "detail" | "record">(
    () => readRoute().view
  )
  const [pairId, setPairId] = useState(() => readRoute().pair)
  const [jobId, setJobId] = useState(() => readRoute().run)
  const [tab, setTab] = useState(() => readRoute().tab)
  const [uploadOpen, setUploadOpen] = useState(false)
  const [connectionOpen, setConnectionOpen] = useState(false)
  const [trainTake, setTrainTake] = useState<Capture | null>(null)
  const [deleteTake, setDeleteTake] = useState<Capture | null>(null)
  const [recordName, setRecordName] = useState("")
  const [libraryVersion, setLibraryVersion] = useState(0)
  const take = studio.state.captures.find((item) => item.id === pairId)
  const jobs = studio.state.jobs.filter((item) => item.capture_id === pairId)
  const job =
    jobs.find((item) => item.id === jobId) || jobs.find(active) || jobs[0]
  useEffect(() => {
    const changed = () => {
      const route = readRoute()
      setView(route.view)
      setPairId(route.pair)
      setJobId(route.run)
      setTab(route.tab)
    }
    window.addEventListener("hashchange", changed)
    return () => window.removeEventListener("hashchange", changed)
  }, [])
  function library() {
    setView("library")
    location.hash = "library"
  }
  function selectRun(id: string) {
    setJobId(id)
    history.replaceState(null, "", "#pair/" + pairId + "/" + id + "/" + tab)
  }
  function selectTab(value: string) {
    setTab(value)
    history.replaceState(
      null,
      "",
      "#pair/" + pairId + "/" + (job?.id || "") + "/" + value
    )
  }
  function open(id: string, run?: string) {
    setPairId(id)
    setJobId(run || "")
    setTab("training")
    setView("detail")
    window.scrollTo({ top: 0 })
    location.hash = "pair/" + id + "/" + (run || "") + "/training"
  }
  return (
    <div className="min-h-svh">
      <a
        href="#main"
        onClick={(event) => {
          event.preventDefault()
          document.getElementById("main")?.focus()
        }}
        className="sr-only focus:not-sr-only focus:fixed focus:top-2 focus:left-2 focus:z-50 focus:rounded-lg focus:bg-primary focus:p-3 focus:text-primary-foreground"
      >
        Skip to content
      </a>
      <header className="border-b bg-card/35">
        <div className="mx-auto flex h-18 max-w-7xl items-center justify-between px-5 sm:px-8 lg:px-10">
          <div className="flex items-center gap-5">
            <a
              href="#library"
              onClick={(event) => {
                event.preventDefault()
                library()
              }}
              aria-label="Neural Amp Modeler home"
              className="flex items-center gap-3"
            >
              <span className="flex size-8 items-center justify-center rounded-lg border border-primary/25 bg-primary/10 text-primary">
                <AudioLines className="size-4.5" />
              </span>
              <span className="text-sm font-medium tracking-tight sm:text-base">
                Neural Amp Modeler
              </span>
            </a>
            <span className="hidden items-center gap-4 text-xs text-muted-foreground md:flex">
              <span className="h-5 w-px rotate-15 bg-border" />
              {view === "library" ? (
                "Library"
              ) : (
                <>
                  <button onClick={library} className="hover:text-foreground">
                    Library
                  </button>
                  <ChevronRight className="size-3" />
                  <span className="max-w-50 truncate">
                    {view === "record" ? "Record" : take?.name}
                  </span>
                </>
              )}
            </span>
          </div>
          <Button
            variant="ghost"
            size="icon-sm"
            onClick={toggle}
            aria-label={
              theme === "dark"
                ? "Switch to light theme"
                : "Switch to dark theme"
            }
            className="text-muted-foreground"
          >
            {theme === "dark" ? (
              <Sun className="size-4" />
            ) : (
              <Moon className="size-4" />
            )}
          </Button>
        </div>
      </header>
      <main
        id="main"
        tabIndex={-1}
        className="mx-auto max-w-7xl px-5 pt-10 pb-20 sm:px-8 sm:pt-12 lg:px-10"
      >
        {studio.offline && (
          <div
            role="alert"
            className="mb-6 flex items-center justify-between gap-4 rounded-lg border border-destructive/20 bg-destructive/5 px-4 py-3 text-sm"
          >
            <span className="flex items-center gap-2">
              <CircleAlert className="size-4 shrink-0 text-destructive" />
              {studio.offline}
            </span>
            <Button
              size="sm"
              variant="outline"
              onClick={() =>
                studio.config ? void studio.refresh() : void studio.initialize()
              }
            >
              Retry
            </Button>
          </div>
        )}
        {studio.state.bridge.recording && view !== "record" && (
          <div className="mb-6 flex items-center justify-between gap-4 rounded-lg border border-primary/20 bg-primary/5 px-4 py-3 text-sm">
            <span className="flex items-center gap-2">
              <span className="size-2 animate-pulse rounded-full bg-primary" />
              A live recording is in progress.
            </span>
            <Button
              variant="outline"
              size="sm"
              onClick={() => {
                setView("record")
                location.hash = "record"
              }}
            >
              View recording
            </Button>
          </div>
        )}
        {view === "detail" && take ? (
          <Suspense
            fallback={
              <div className="py-16 text-sm text-muted-foreground">
                Loading training results…
              </div>
            }
          >
            <PairDetail
              take={take}
              jobs={jobs}
              selectedJob={job}
              onSelectJob={selectRun}
              tab={tab}
              onTab={selectTab}
              onBack={library}
              onTrain={() => setTrainTake(take)}
            />
          </Suspense>
        ) : view === "record" ? (
          <Recording
            studio={studio}
            initialName={recordName}
            onBack={library}
            onConnection={() => setConnectionOpen(true)}
          />
        ) : (
          <Library
            key={libraryVersion}
            studio={studio}
            onAdd={() => setUploadOpen(true)}
            onOpen={open}
            onTrain={setTrainTake}
            onDelete={setDeleteTake}
          />
        )}
      </main>
      {uploadOpen && (
        <UploadDialog
          studio={studio}
          onClose={() => setUploadOpen(false)}
          onAdded={() => {
            library()
            setLibraryVersion((value) => value + 1)
          }}
          onRecord={(name) => {
            setRecordName(name)
            setUploadOpen(false)
            setView("record")
            location.hash = "record"
          }}
        />
      )}
      {trainTake && (
        <TrainDialog
          take={trainTake}
          studio={studio}
          onClose={() => setTrainTake(null)}
          onStarted={(next) => open(next.capture_id, next.id)}
        />
      )}
      {deleteTake && (
        <DeleteDialog
          take={deleteTake}
          studio={studio}
          onClose={() => setDeleteTake(null)}
        />
      )}
      {connectionOpen && (
        <ConnectionDialog
          studio={studio}
          onClose={() => setConnectionOpen(false)}
        />
      )}
    </div>
  )
}
