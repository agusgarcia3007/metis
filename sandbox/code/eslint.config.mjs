import eslint from "@eslint/js";
import tseslint from "typescript-eslint";

export default tseslint.config(
  {
    ignores: [
      "**/.git/**",
      "**/coverage/**",
      "**/dist/**",
      "**/node_modules/**",
      "**/target/**",
      "**/*.config.*",
    ],
  },
  eslint.configs.recommended,
  ...tseslint.configs.recommended,
  {
    languageOptions: {
      globals: {
        Buffer: "readonly",
        console: "readonly",
        describe: "readonly",
        expect: "readonly",
        it: "readonly",
        process: "readonly",
        test: "readonly",
      },
    },
    rules: {
      "@typescript-eslint/no-explicit-any": "off",
    },
  },
);
