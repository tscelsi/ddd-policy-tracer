import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Parliamentary Questions",
  description: "Ask questions about parliamentary speeches and bills.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
