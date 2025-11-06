#FROM debian:bullseye
FROM node:18-alpine

COPY package*.json /app/
WORKDIR /app

#RUN apt update || : && apt install curl -y
#RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash -
#RUN apt install -y nodejs 
RUN npm install

COPY . .

RUN npm run build

EXPOSE 3000

CMD ["npm", "run", "start"]
