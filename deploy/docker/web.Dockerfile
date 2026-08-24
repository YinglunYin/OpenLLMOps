FROM node:22-alpine AS frontend-build

WORKDIR /src
ARG VITE_API_BASE_URL=/api
ARG VITE_USE_MOCKS=false
ENV VITE_API_BASE_URL=${VITE_API_BASE_URL} \
    VITE_USE_MOCKS=${VITE_USE_MOCKS}
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM nginx:1.28-alpine

RUN apk add --no-cache openssl

COPY deploy/nginx/nginx.conf /etc/nginx/nginx.conf
COPY deploy/nginx/10-ensure-tls.sh /docker-entrypoint.d/10-ensure-tls.sh
COPY --from=frontend-build /src/dist /usr/share/nginx/html

RUN chmod 0555 /docker-entrypoint.d/10-ensure-tls.sh \
    && rm -f /etc/nginx/conf.d/default.conf \
    && mkdir -p /etc/nginx/tls \
    && chown nginx:nginx /etc/nginx/tls

EXPOSE 8080 8443
