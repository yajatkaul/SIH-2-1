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
      <div className="h-screen pt-[20px] pl-[70px] pr-[70px]">
        <div className="h-[80px] flex justify-between items-center border border-[#6d6d6d] rounded-[40px] pl-[40px] pr-[10px]">
          <div className="flex text-[30px] font-bold">
            <p className="text-black">Vani</p>
            <p className="text-orange-500">Sutra</p>
          </div>
          <div className="flex text-black gap-[20px] font-bold h-full justify-center items-center pt-[10px] pb-[10px]">
            <p className="cursor-pointer">Home</p>
            <p className="cursor-pointer">About Us</p>
            <p className="cursor-pointer">Features</p>
            <p className="cursor-pointer">Services</p>
            <p className="bg-black text-white rounded-[50px] h-full flex items-center w-[140px] justify-center cursor-pointer">
              Contact
            </p>
          </div>
        </div>

        <div className="flex justify-center gap-[90px] items-center mt-[30px] mb-[90px]">
          <div>
            <div className="flex flex-col text-black">
              <p className="text-orange-600 text-[50px]">Ensure Safety by</p>
              <p className="text-[50px] mb-[10px]">Audio Spotting</p>
              <p>Real-time key word spotting for audio files,</p>
              <p className="mb-[30px]">input your audio here</p>
              <button className="bg-black h-[70px] rounded-[30px] text-white text-[25px] font-bold">
                Input Here
              </button>
            </div>
          </div>
          <div>
            <img src="/homeimg.png" alt="" className="max-h-[700px]" />
          </div>
        </div>
      </div>

      <div className="pl-[70px] pr-[70px] text-black flex justify-center items-center flex-col gap-[30px] mb-[50px]">
        <div className="flex gap-8 items-center">
          <div className="flex flex-col">
            <div className="flex gap-2">
              <p className="text-[50px]">About</p>
              <p className="text-[50px] text-orange-600">Us</p>
            </div>
            <div className="flex flex-col text-[20px]">
              <p>At VaniSutra, we are dedicated to harnessing the</p>
              <p>power of cutting-edge technology to enhance</p>
              <p>security and communication in critical situations. Our</p>
              <p>innovative audio analysis system combines</p>
              <p>advanced preprocessing techniques, robust speech</p>
              <p>recognition, and state-of-the-art NLP models to</p>
              <p>provide real-time insights that can make a difference </p>
              <p>in emergencies like kidnappings or interrogations.</p>
            </div>
          </div>

          <img src="/aboutus1.png" alt="" className="max-h-[400px]" />
        </div>

        <div className="flex gap-8 items-center">
          <img src="/aboutus2.png" alt="" className="max-h-[300px]" />

          <div className="flex flex-col">
            <div className="flex gap-2">
              <p className="text-[50px]">Our</p>
              <p className="text-[50px] text-orange-600">Mission</p>
            </div>
            <div className="flex flex-col text-[20px]">
              <p>At VaniSutra, we are dedicated to harnessing the</p>
              <p>power of cutting-edge technology to enhance</p>
              <p>security and communication in critical situations. Our</p>
              <p>innovative audio analysis system combines</p>
              <p>advanced preprocessing techniques, robust speech</p>
              <p>recognition, and state-of-the-art NLP models to</p>
              <p>provide real-time insights that can make a difference </p>
              <p>in emergencies like kidnappings or interrogations.</p>
            </div>
          </div>
        </div>
      </div>

      <div className="pl-[70px] pr-[70px] text-black flex justify-center items-center flex-col gap-[30px] pb-[50px]">
        <div className="flex text-[50px] font-bold">
          <p>Feat</p>
          <p className="text-orange-600">ures</p>
        </div>

        <div className="flex text-black w-full flex-wrap gap-[80px] justify-center">
          <div className="w-[600px] flex flex-col shadow-md shadow-black p-[10px] rounded-[30px]">
            <div className="h-full flex justify-center items-center">
              <img src="/feature1.png" alt="" />
            </div>

            <p className="text-[#A0674E] text-[30px] flex justify-center">
              Language-Agnostic Keyword Spotting
            </p>
            <div className="flex flex-col text-[20px] mt-[10px]">
              <p className="text-center">
                The system identifies keywords across multiple languages using
                advanced models like Wav2Vec 2.0 and BERT, ensuring accurate
                keyword detection without relying on language-specific features.
              </p>
            </div>
          </div>

          <div className="w-[600px] flex flex-col shadow-md shadow-black p-[10px] rounded-[30px]">
            <div className="h-full flex justify-center items-center">
              <img src="/feature2.png" alt="" />
            </div>

            <p className="text-[#A0674E] text-[30px] flex justify-center">
              Context-Aware Detection
            </p>
            <div className="flex flex-col text-[20px] mt-[10px]">
              <p className="text-center">
                The system enhances keyword detection by incorporating
                contextual cues such as time of day, user behavior, and
                location, ensuring more relevant and precise keyword
                identification.
              </p>
            </div>
          </div>

          <div className="w-[600px] flex flex-col shadow-md shadow-black p-[10px] rounded-[30px]">
            <div className="h-full flex justify-center items-center">
              <img src="/feature3.png" alt="" />
            </div>

            <p className="text-[#A0674E] text-[30px] flex justify-center">
              Real-Time Alerts and Monitoring
            </p>
            <div className="flex flex-col text-[20px] mt-[10px]">
              <p className="text-center">
                Integrated real-time alerts notify users of any mismatches or
                errors in the extracted data, while Flask and Prometheus enable
                ongoing system performance monitoring and visualization for
                efficient operations.
              </p>
            </div>
          </div>

          <div className="w-[600px] flex flex-col shadow-md shadow-black p-[10px] rounded-[30px]">
            <div className="h-full flex justify-center items-center">
              <img src="/feature4.png" alt="" />
            </div>

            <p className="text-[#A0674E] text-[30px] flex justify-center">
              Real-Time Speech-to-Text Conversion
            </p>
            <div className="flex flex-col text-[20px] mt-[10px]">
              <p className="text-center">
                Audio is converted into text in real-time using tools like
                Google Speech-to-Text or DeepSpeech, allowing for immediate
                transcription and analysis of spoken content.
              </p>
            </div>
          </div>
        </div>
      </div>

      <div className="pl-[70px] pr-[70px] text-black flex justify-center items-center flex-col gap-[30px] pb-[50px]">
        <div className="flex text-[50px] font-bold">
          <p>Serv</p>
          <p className="text-orange-600">ices</p>
        </div>

        <div className="flex text-black w-full flex-wrap gap-[80px] justify-center">
          <div className="w-[600px] flex flex-col shadow-md shadow-black p-[10px] rounded-[30px]">
            <div className="h-full flex justify-center items-center">
              <img src="/service1.png" alt="" />
            </div>

            <p className="text-[#A0674E] text-[30px] flex justify-center">
              Tailored Speech Recognition Solutions
            </p>
            <div className="flex flex-col text-[20px] mt-[10px]">
              <p className="text-center">
                We design and implement customized speech-to-text systems for
                organizations, enabling seamless conversion of spoken language
                into accurate, actionable data tailored to specific industries
                like legal, healthcare, and customer service.
              </p>
            </div>
          </div>

          <div className="w-[600px] flex flex-col shadow-md shadow-black p-[10px] rounded-[30px]">
            <div className="h-full flex justify-center items-center">
              <img src="/service2.png" alt="" />
            </div>

            <p className="text-[#A0674E] text-[30px] flex justify-center">
              Keyword Monitoring & Compliance Audits
            </p>
            <div className="flex flex-col text-[20px] mt-[10px]">
              <p className="text-center">
                Our service offers real-time keyword monitoring to help
                organizations track and analyze conversations for compliance,
                security, or quality assurance purposes, ensuring adherence to
                industry regulations and internal standards.
              </p>
            </div>
          </div>

          <div className="w-[600px] flex flex-col shadow-md shadow-black p-[10px] rounded-[30px]">
            <div className="h-full flex justify-center items-center">
              <img src="/service3.png" alt="" />
            </div>

            <p className="text-[#A0674E] text-[30px] flex justify-center">
              Enterprise-Grade Audio Analytics Integration
            </p>
            <div className="flex flex-col text-[20px] mt-[10px]">
              <p className="text-center">
                We integrate advanced audio analytics into your existing
                platforms, providing detailed insights from voice data that
                enhance decision-making and operational efficiency, while
                ensuring compatibility with your existing tools and
                infrastructure.
              </p>
            </div>
          </div>

          <div className="w-[600px] flex flex-col shadow-md shadow-black p-[10px] rounded-[30px]">
            <div className="h-full flex justify-center items-center">
              <img src="/service4.png" alt="" />
            </div>

            <p className="text-[#A0674E] text-[30px] flex justify-center">
              24/7 Real-Time Alerting & Support
            </p>
            <div className="flex flex-col text-[20px] mt-[10px]">
              <p className="text-center">
                We offer continuous, real-time alert systems for critical
                keyword detection or discrepancies, alongside 24/7 support to
                ensure your organization’s systems are always optimized and
                functional.
              </p>
            </div>
          </div>
        </div>
      </div>

      <div className="bg-[#D0CFCE2E] text-black pl-[160px] pr-[160px] pb-[20px] pt-[20px]">
        <div className="flex gap-2 text-[50px] justify-center">
          <div className="flex">
            <p className="text-orange-600">F</p>
            <p>requently</p>
          </div>
          <div className="flex">
            <p className="text-orange-600">A</p>
            <p>sked</p>
          </div>
          <div className="flex">
            <p className="text-orange-600">Q</p>
            <p>uestions</p>
          </div>
        </div>

        <div className="flex flex-col gap-[20px]">
          <div
            tabIndex={0}
            className="collapse collapse-plus border-base-300 bg-white border text-black"
          >
            <div className="collapse-title text-xl font-medium">
              How will it help us?
            </div>
            <div className="collapse-content">
              <p>Placeholder</p>
            </div>
          </div>
          <div
            tabIndex={0}
            className="collapse collapse-plus border-base-300 bg-white border text-black"
          >
            <div className="collapse-title text-xl font-medium">
              Who all are using it ?
            </div>
            <div className="collapse-content">
              <p>Placeholder</p>
            </div>
          </div>
          <div
            tabIndex={0}
            className="collapse collapse-plus border-base-300 bg-white border text-black"
          >
            <div className="collapse-title text-xl font-medium">
              What is the tech stack behind it ?
            </div>
            <div className="collapse-content">
              <p>Placeholder</p>
            </div>
          </div>
          <div
            tabIndex={0}
            className="collapse collapse-plus border-base-300 bg-white border text-black"
          >
            <div className="collapse-title text-xl font-medium">
              Why we need this kind of system ?
            </div>
            <div className="collapse-content">
              <p>Placeholder</p>
            </div>
          </div>
          <div
            tabIndex={0}
            className="collapse collapse-plus border-base-300 bg-white border text-black"
          >
            <div className="collapse-title text-xl font-medium">
              What is the purpose of this system ?
            </div>
            <div className="collapse-content">
              <p>Placeholder</p>
            </div>
          </div>
        </div>
      </div>

      <div className="flex flex-col text-black">
        <div className="flex p-[100px] gap-[100px] justify-center">
          <div className="flex flex-col">
            <div className="flex text-[40px]">
              <p>Contact</p>
              <p className="text-orange-600">Us</p>
            </div>
            <div className="text-[20px]">
              <p>Address</p>
              <hr className="border-orange-600" />
              <p className="max-w-[300px] mt-[10px] mb-[10px] text-wrap">
                House no 164-A ,street no 2 near lavanya hospital,maidan garhi
                ext chattarpur,New Delhi 110070
              </p>
              <hr className="border-orange-600" />
              <p>Contact No +91 882147754</p>
              <p>E-mail : vanisutra10@gmail.com</p>
              <div className="flex gap-[10px] mt-[20px] ml-[20px]">
                <img
                  src="/facebook.png"
                  alt=""
                  className="max-h-[60px] cursor-pointer"
                />
                <img
                  src="/linkedin.png"
                  alt=""
                  className="max-h-[60px] cursor-pointer"
                />
                <img
                  src="/twitter.png"
                  alt=""
                  className="max-h-[60px] cursor-pointer"
                />
              </div>
            </div>
          </div>

          <div className="flex-col">
            <p className="text-[30px]">Get in touch</p>
            <div className="flex flex-col gap-2">
              <div>
                <div className="flex gap-2">
                  <input
                    type="text"
                    placeholder="Your Name"
                    className="input input-bordered"
                  />
                  <input
                    type="text"
                    placeholder="Email"
                    className="input input-bordered"
                  />
                </div>
              </div>

              <div>
                <div className="flex gap-2">
                  <input
                    type="text"
                    placeholder="Phone no"
                    className="input input-bordered"
                  />
                  <input
                    type="text"
                    placeholder="Gender"
                    className="input input-bordered"
                  />
                </div>
              </div>

              <div>
                <textarea
                  className="textarea textarea-bordered w-full"
                  placeholder="Message"
                ></textarea>
              </div>

              <div>
                <button className="bg-black text-orange-600 font-bold h-[60px] w-[300px] rounded-[20px] text-[30px]">
                  Submit
                </button>
              </div>
            </div>
          </div>
        </div>

        <div className="bg-black flex justify-center items-center h-[80px] text-[30px] font-bold text-white">
          All Right Reserved @vanisutra
        </div>
      </div>
    </div>
  );
}
