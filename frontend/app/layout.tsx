import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'XML-RoBERTa 학습 및 추론 시스템',
  description: 'XML-RoBERTa 모델을 사용한 텍스트 분류 및 학습 시스템',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="ko">
      <body className="min-h-screen bg-gray-50">
        <div className="container mx-auto px-4 py-8">
          {children}
        </div>
      </body>
    </html>
  );
}
