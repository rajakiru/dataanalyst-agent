import { NextRequest, NextResponse } from "next/server";

const BACKEND = process.env.BACKEND_URL ?? "http://localhost:8000";

export async function POST(req: NextRequest) {
  const body = await req.formData();
  const res = await fetch(`${BACKEND}/plan`, { method: "POST", body });
  const data = await res.json();
  return NextResponse.json(data);
}
