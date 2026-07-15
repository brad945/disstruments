import "./globals.css";

export const metadata = { title: "Disstruments", description: "Song deconstruction" };

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <div className="topbar">
          <a href="/" className="logo">DISSTRUMENTS</a>
          <span className="sub">deconstruct · learn · produce</span>
        </div>
        {children}
      </body>
    </html>
  );
}
