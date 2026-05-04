FROM node:20-alpine
WORKDIR /app
COPY pakage*.json ./
RUN npm install
COPY . .
EXPOSE 3000
CMD ["node", "src/app.js"]