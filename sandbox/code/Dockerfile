FROM node:22.17.0-bookworm-slim@sha256:b04ce4ae4e95b522112c2e5c52f781471a5cbc3b594527bcddedee9bc48c03a0

RUN apt-get update \
    && apt-get install -y --no-install-recommends git gosu \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/metis-tools
COPY package.json package-lock.json /opt/metis-tools/
RUN npm ci --ignore-scripts --no-audit --no-fund

COPY eslint.config.mjs metis-ts-parse.mjs entrypoint.sh wrappers/ /opt/metis-tools/
RUN chmod 0555 \
    /opt/metis-tools/entrypoint.sh \
    /opt/metis-tools/metis-ts-parse.mjs \
    /opt/metis-tools/metis-ts-lint \
    /opt/metis-tools/metis-ts-typecheck \
    /opt/metis-tools/metis-vitest \
    && ln -s /opt/metis-tools/metis-ts-parse.mjs /usr/local/bin/metis-ts-parse \
    && ln -s /opt/metis-tools/metis-ts-lint /usr/local/bin/metis-ts-lint \
    && ln -s /opt/metis-tools/metis-ts-typecheck /usr/local/bin/metis-ts-typecheck \
    && ln -s /opt/metis-tools/metis-vitest /usr/local/bin/metis-vitest

ENV PATH="/workspace/node_modules/.bin:/opt/metis-tools/node_modules/.bin:/usr/local/bin:/usr/bin:/bin"

ENTRYPOINT ["/opt/metis-tools/entrypoint.sh"]
