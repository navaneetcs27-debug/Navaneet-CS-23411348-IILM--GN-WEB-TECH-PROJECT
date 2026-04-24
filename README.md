# Traffic Management System

This project can be deployed as a single Flask web app so the frontend and API run from one public URL.

## Deploy on Render

1. Create a GitHub repository and upload this project.
2. Sign in to Render and create a new `Web Service`.
3. Connect your GitHub repository.
4. Render should detect `render.yaml` automatically.
5. Deploy the service.
6. After deployment, open the public Render URL and test:
   - `/`
   - `/login.html`
   - `/index.html`
   - `/healthz`

## Default login

- Username: `admin`
- Password: `1234`

## Search visibility

After deployment, update these files with your real public domain:

- `robots.txt`
- `sitemap.xml`

Then:

1. Add a custom domain in Render if you want a cleaner website address.
2. Submit your `sitemap.xml` in Google Search Console.
3. Wait for Google indexing. Search appearance is not instant.

## Notes

- `Road2`, `Road3`, and `Road4` video files are not present, so live traffic API data will return `0` for those roads unless you add the missing videos.
- YOLO detection routes require model files inside the `yolo` folder.
