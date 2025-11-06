import Head from "next/head";
import ImageUploader from "../components/ImageUploader";

export default function Home() {
  return (
    <>
      <Head>
        <title>SpiderNets: Estimating Fear Ratings</title>
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <meta
          name="description"
          content="Minimal demo: title and image upload."
        />
      </Head>
      <main className="container">
        <h1 className="title">SpiderNets: Estimating Fear Ratings</h1>
        <p className="subtitle">
          Upload an image to preview. Inference disabled.
        </p>
        <ImageUploader />
      </main>
    </>
  );
}
