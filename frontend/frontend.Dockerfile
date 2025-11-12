# Stage 1: Build the Next.js Application
FROM node:18-alpine AS builder
WORKDIR /app

COPY package.json package-lock.json ./
RUN npm install

COPY . .
RUN npm run build

# ---
# Stage 2: Serve the Application (Minimal Runtime)
FROM node:18-alpine
WORKDIR /app

ENV NODE_ENV production
ENV NEXT_TELEMETRY_DISABLED 1

COPY --from=builder /app/.next ./.next
COPY --from=builder /app/public ./public

COPY --from=builder /app/package.json ./package.json
RUN npm install --production

CMD ["npm", "start"]
EXPOSE 3000