//@ts-nocheck
"use client";
import { useKeyWords } from "@/hooks/getKeywords";
import { useGSAP } from "@gsap/react";
import gsap from "gsap";
import { ScrollTrigger } from "gsap/ScrollTrigger";
import { FileAudio } from "lucide-react";
import { useEffect, useState } from "react";

gsap.registerPlugin(ScrollTrigger);

export default function Home() {
  function extractKeywords(responseText: string) {
    if (!responseText) return;
    const result = [];
    const lines = responseText.split("\n");
    const bulletPoints = lines.filter((line) => line.trim().startsWith("*"));
    const cleanedBulletPoints = bulletPoints.map((point) =>
      point.replace("*", "").trim()
    );
    result.push(...cleanedBulletPoints);

    return result;
  }

  useGSAP(() => {
    gsap.from("#LogoHeader", {
      opacity: 0,
      y: -100,
      duration: 2,
    });

    gsap.from("#navLinks p", {
      opacity: 0,
      duration: 2,
      stagger: 0.3,
      y: -100,
    });

    gsap.from("#Logo", {
      scale: 0,
      opacity: 0,
      rotate: 360,
      duration: 2,
      scrollTrigger: {
        start: "top 0%",
        end: "top -10%",
        scrub: 2,
      },
    });

    gsap.from("#page1 p", {
      opacity: 0,
      x: 500,
      scrollTrigger: {
        trigger: "#page1",
        start: "top 100%",
        end: "top -40%",
        scrub: 2,
      },
    });

    gsap.from("#page1 img", {
      opacity: 0,
      x: -500,
      scrollTrigger: {
        trigger: "#page1",
        start: "top 100%",
        end: "top -40%",
        scrub: 4,
      },
    });

    gsap.from("#page2", {
      opacity: 0,
      x: -100,
      duration: 1,
      stagger: 0.3,
      scrollTrigger: {
        trigger: "#page2",
        start: "top 100%",
        end: "top 30%",
        scrub: 4,
      },
    });
  });

  const [file, setFile] = useState();
  const { loading, send, data } = useKeyWords();
  const [formatData, setFormatData] = useState();

  useEffect(() => {
    setFormatData(extractKeywords(data));
  }, [data]);

  return (
    <div>
      <div className="md:flex h-[90px] justify-between hidden">
        <img
          src="/LogoWithName.png"
          alt=""
          className="cursor-pointer"
          id="LogoHeader"
        />
        <div className="flex gap-4 items-center" id="navLinks">
          <p
            className="cursor-pointer font-bold"
            onClick={() =>
              document
                .getElementById("nav1")
                .scrollIntoView({ behavior: "smooth" })
            }
          >
            About us
          </p>
          <p
            className="cursor-pointer font-bold"
            onClick={() =>
              document
                .getElementById("nav2")
                .scrollIntoView({ behavior: "smooth" })
            }
          >
            Key features
          </p>
          <p
            className="cursor-pointer font-bold"
            onClick={() =>
              document
                .getElementById("nav3")
                .scrollIntoView({ behavior: "smooth" })
            }
          >
            Concerned Departments
          </p>
        </div>
      </div>

      <div className="flex h-screen items-center justify-center">
        <img src="/Logo.png" alt="" id="Logo" />
      </div>

      <div
        className="min-h-screen flex bg-[#161616] flex-col overflow-hidden"
        id="page1"
      >
        <p
          className="flex justify-center w-full text-[40px] font-bold mb-[50px]"
          id="nav1"
        >
          About Us
        </p>
        <div className="flex text-[20px] font-bold gap-8 justify-center flex-col items-center">
          <div className="flex gap-8 md:pr-[300px] pr-[0px] md:flex-row flex-col">
            <img src="/img1.png" alt="" className="flex w-[250px] h-[240px] " />
            <p className="w-[300px]">
              VaniSutra uses pre-trained audio feature extraction with Few-Shot
              models like Prototypical Networks to detect keywords with minimal
              examples.
            </p>
          </div>

          <div className="flex gap-8 md:pl-[300px] pl-[0px] md:flex-row flex-col">
            <img src="/img2.png" alt="" className="flex w-[200px] h-[240px]" />
            <p className="w-[300px]">
              It handles various sample rates (8kHz-48kHz) with audio
              augmentation, optimizing performance using efficient architectures
              like MobileNet and model compression.
            </p>
          </div>

          <div className="flex gap-8 md:pr-[300px] pr-[0px] md:flex-row flex-col">
            <img src="/img3.png" alt="" className="flex w-[200px] h-[240px]" />
            <p className="w-[300px]">
              Keywords are localized with sliding windows or attention, and the
              modular design allows easy addition of new keywords. The system
              focuses on high accuracy, low latency, and scalability.
            </p>
          </div>
        </div>
      </div>

      <div
        className="min-h-screen flex flex-col bg-[#242424] items-center"
        id="page2"
      >
        <p
          className="flex justify-center w-full text-[40px] font-bold"
          id="nav2"
        >
          Demo
        </p>
        <p className="p-[10px]">
          Innovative/ Unique Feature Context-aware Keyword Spotting: Introduce a
          context engine that enhances keyword detection based on situational
          cues like time of day, location, or user behavior, further refining
          the accuracy and relevancy of keyword identification.
        </p>
        <div className="flex flex-col justify-center items-center gap-4">
          <div
            className="bg-[#292828] w-[300px] h-[80px] md:w-[400px] flex justify-center items-center flex-col rounded-[30px] hover:cursor-pointer hover:bg-[#222020]"
            onClick={() => document.getElementById("file_input")?.click()}
          >
            Input Audio
            <FileAudio />
          </div>
          {loading ? (
            <button className="bg-[#292828] p-[10px] rounded-[30px] text-[20px] font-bold h-[60px] w-[100px] hover:bg-[#222020]">
              <span className="loading loading-ring loading-lg"></span>
            </button>
          ) : (
            <button
              className="bg-[#292828] p-[10px] rounded-[30px] text-[20px] font-bold h-[60px] w-[100px] hover:bg-[#222020]"
              onClick={() => {
                send(file);
              }}
            >
              Submit
            </button>
          )}

          <input
            type="file"
            className="hidden"
            id="file_input"
            onChange={(e) => {
              setFile(e.target.files[0]);
            }}
          />
        </div>

        <div className="flex flex-col items-center justify-center">
          {formatData ? (
            <p className="text-[50px] font-bold">
              Potential keywords found are:
            </p>
          ) : (
            ""
          )}
          {formatData?.map((data, index) => {
            return (
              <p
                key={index}
                className="flex items-center justify-center font-bold text-[30px]"
              >
                - {data}
              </p>
            );
          })}
        </div>
      </div>

      <div className="flex flex-col bg-[#000000]">
        <p
          className="flex justify-center w-full md:text-[40px] font-bold text-[25px]"
          id="nav3"
        >
          Concerned Departments
        </p>
        <div className="overflow-hidden bg-[#242424]/40 backdrop-blur-md">
          <div className="flex items-center gap-4 animate-scroll">
            <img src="/NTRO.png" alt="" className="h-[350px]" />
            <img
              src="/DRDO.png"
              alt=""
              className="h-[300px] w-[280px] object-contain"
            />
            <img
              src="/CBI.png"
              alt=""
              className="h-[300px] w-[280px] object-contain"
            />
            <img src="/NSA.png" alt="" className="h-[300px]" />
            <img src="/FBI.png" alt="" className="h-[300px]" />
            <img
              src="/FCC.png"
              alt=""
              className="h-[300px] w-[280px] object-contain"
            />
            <img src="/NDRF.png" alt="" className="h-[300px]" />
            {}
            <img src="/NTRO.png" alt="" className="h-[350px]" />
            <img
              src="/DRDO.png"
              alt=""
              className="h-[300px] w-[280px] object-contain"
            />
            <img
              src="/CBI.png"
              alt=""
              className="h-[300px] w-[280px] object-contain"
            />
            <img src="/NSA.png" alt="" className="h-[300px]" />
            <img src="/FBI.png" alt="" className="h-[300px]" />
            <img
              src="/FCC.png"
              alt=""
              className="h-[300px] w-[280px] object-contain"
            />
            <img src="/NDRF.png" alt="" className="h-[300px]" />
            {}
            <img src="/NTRO.png" alt="" className="h-[350px]" />
            <img
              src="/DRDO.png"
              alt=""
              className="h-[300px] w-[280px] object-contain"
            />
            <img
              src="/CBI.png"
              alt=""
              className="h-[300px] w-[280px] object-contain"
            />
            <img src="/NSA.png" alt="" className="h-[300px]" />
            <img src="/FBI.png" alt="" className="h-[300px]" />
            <img
              src="/FCC.png"
              alt=""
              className="h-[300px] w-[280px] object-contain"
            />
            <img src="/NDRF.png" alt="" className="h-[300px]" />
            {}
            <img src="/NTRO.png" alt="" className="h-[350px]" />
            <img
              src="/DRDO.png"
              alt=""
              className="h-[300px] w-[280px] object-contain"
            />
            <img
              src="/CBI.png"
              alt=""
              className="h-[300px] w-[280px] object-contain"
            />
            <img src="/NSA.png" alt="" className="h-[300px]" />
            <img src="/FBI.png" alt="" className="h-[300px]" />
            <img
              src="/FCC.png"
              alt=""
              className="h-[300px] w-[280px] object-contain"
            />
            <img src="/NDRF.png" alt="" className="h-[300px]" />
          </div>
        </div>
      </div>
    </div>
  );
}
