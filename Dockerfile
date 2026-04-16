FROM nginx:alpine

RUN sed -i \
    -e 's/listen\s*80;/listen 3000;/g' \
    -e 's/listen\s*\[::\]:80;/listen [::]:3000;/g' \
    -e 's/index\s*index\.html\s*index\.htm\s*;/index se-guard-dashboard.html index.html index.htm;/g' \
    /etc/nginx/conf.d/default.conf

COPY Project/frontend/ /usr/share/nginx/html/

EXPOSE 3000

CMD ["nginx", "-g", "daemon off;"]
