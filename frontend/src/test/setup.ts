import '@testing-library/jest-dom/vitest'
import { cleanup } from '@testing-library/react'
import { afterEach } from 'vitest'

// `globals: false` means Testing Library cannot auto-register its cleanup hook,
// so unmount between tests explicitly. Without this, repeated `render(<App />)`
// calls stack duplicate DOM and every getByRole/getByLabelText becomes ambiguous.
afterEach(cleanup)
