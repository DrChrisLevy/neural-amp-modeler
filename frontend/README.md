# Neural Amp Modeler UI

React, TypeScript, and shadcn/ui. See the [root README](../README.md#frontend-development) for development, build, and deployment commands.

- `src/components/`: library, training, recording, and synchronized comparison views.
- `src/components/ui/`: shadcn/ui components, generated with the Radix Nova preset.
- `src/hooks/use-studio.ts`: API state, live monitoring, and independent background uploads.
- `src/lib/engines/`: tested audio playback, upload, and training-metric logic shared with the previous UI.

Add another shadcn component with `npx shadcn@latest add <component>` from this directory. Generated build assets go into the Python package's `static/` directory and are not committed.
