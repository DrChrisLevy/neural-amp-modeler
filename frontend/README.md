# Neural Amp Modeler UI

React, TypeScript, and shadcn/ui. See the [root README](../README.md#run-your-own-app) for deployment and the upload/train/download workflow.

## Local development

Node.js is only needed for local UI development; Modal builds the frontend automatically during deployment. For local development, use Node.js 22.12+ (24 recommended). From the repository root:

```sh
uv sync
npm ci --prefix frontend
npm run build --prefix frontend
uv run neural-amp-modeler serve
```

In a second terminal, run `npm run dev --prefix frontend` and open the Vite URL. API and live monitoring requests proxy to the Python server on port 8000. Local serving supports UI development; training requires the Modal deployment's worker.

## Source

- `src/components/`: library, training, recording, and synchronized comparison views.
- `src/components/ui/`: shadcn/ui components, generated with the Radix Nova preset.
- `src/hooks/use-studio.ts`: API state, live monitoring, and independent background uploads.
- `src/lib/engines/`: tested audio playback, upload, and training-metric logic shared with the previous UI.

Add another shadcn component with `npx shadcn@latest add <component>` from this directory. Generated build assets go into the Python package's `static/` directory and are not committed.

## Checks

From the repository root:

```sh
npm run lint --prefix frontend
npm run build --prefix frontend
node --test tests/test_*.cjs
```
