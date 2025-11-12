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
        <div
          className="subtitle"
          style={{
            display: "flex",
            justifyContent: "flex-start",
            alignItems: "center",
            flexDirection: "column",
          }}
        >
          <div>
            <b>Paper:</b> SpiderNets: Estimating Fear Ratings of Spider-Related
            Images with Vision Models{" "}
          </div>
          <div>
            <b>Authors:</b> D. Pegler, D. Steyrl, M. Zhang, A. Karner, J. Arato,
            F. Scharnowski, F. Melinscak{" "}
          </div>
          <div>
            <b>DOI: </b>
            <a href="https://doi.org/10.48550/arXiv.2509.04889" target="_blank">
              10.48550/arXiv.2509.04889
            </a>
          </div>
        </div>
        <ImageUploader />
      </main>
    </>
  );
}
