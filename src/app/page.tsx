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
  const [file, setFile] = useState();
  const { loading, send, data } = useKeyWords();
  const [details, setDetails] = useState({
    yourName: "",
    email: "",
    phoneNo: "",
    gender: "",
    message: "",
  });

  const getKeywords = () => {
    send(file);
  };

  const contactUs = async (e) => {
    e.preventDefault();
    //setResult("Sending....");
    const formData = new FormData(e.target);

    formData.append("access_key", "fe3216da-539f-4d34-9192-bd76918c0623");

    const response = await fetch("https://api.web3forms.com/submit", {
      method: "POST",
      body: formData,
    });

    const data = await response.json();

    if (data.success) {
      //setResult("Form Submitted Successfully");
      e.target.reset();
    } else {
      console.log("Error", data);
      //setResult(data.message);
    }
  };

  return (
    <div>
      <div
        className="pt-[20px] pl-[40px] pr-[40px] md:pl-[70px] md:pr-[70px]"
        id="home"
      >
        <div className="h-[80px] flex justify-between items-center border border-[#6d6d6d] rounded-[40px] md:pl-[40px] md:pr-[10px] overflow-hidden">
          <div className="flex text-[30px] font-bold w-full justify-center items-center md:w-auto">
            <p className="text-black">Vani</p>
            <p className="text-orange-500">Sutra</p>
          </div>
          <div className="md:flex text-black gap-[20px] font-bold h-full justify-center items-center pt-[10px] pb-[10px] hidden">
            <p
              className="cursor-pointer"
              onClick={() =>
                document
                  .getElementById("home")
                  .scrollIntoView({ behavior: "smooth" })
              }
            >
              Home
            </p>
            <p
              className="cursor-pointer"
              onClick={() =>
                document
                  .getElementById("aboutUs")
                  .scrollIntoView({ behavior: "smooth" })
              }
            >
              About Us
            </p>
            <p
              className="cursor-pointer"
              onClick={() =>
                document
                  .getElementById("features")
                  .scrollIntoView({ behavior: "smooth" })
              }
            >
              Features
            </p>
            <p
              className="cursor-pointer"
              onClick={() =>
                document
                  .getElementById("services")
                  .scrollIntoView({ behavior: "smooth" })
              }
            >
              Services
            </p>
            <p
              className="bg-black text-white rounded-[50px] h-full flex items-center w-[140px] justify-center cursor-pointer"
              onClick={() =>
                document
                  .getElementById("contactUs")
                  .scrollIntoView({ behavior: "smooth" })
              }
            >
              Contact Us
            </p>
          </div>
        </div>

        <div
          className="flex justify-center gap-[90px] items-center mt-[30px] mb-[90px] md:flex-row flex-col-reverse"
          id="home"
        >
          <div>
            <div className="flex flex-col text-black">
              <p className="text-orange-600 text-[50px]">Ensure Safety by</p>
              <p className="text-[50px] mb-[10px]">Audio Spotting</p>
              <p>Real-time key word spotting for audio files,</p>
              <p className="mb-[30px]">input your audio here</p>
              <div className="flex flex-col gap-2">
                <button
                  className="bg-black h-[70px] rounded-[30px] text-white text-[25px] font-bold"
                  onClick={() => document.getElementById("fileUpload")?.click()}
                >
                  Input Here
                </button>
                <button
                  className="bg-black h-[70px] rounded-[30px] text-orange-600 text-[25px] font-bold"
                  onClick={() => getKeywords()}
                >
                  Submit
                </button>
                {!data ? "" : <p>This audio is being indicated as: {data}</p>}
              </div>

              <input
                type="file"
                className="hidden"
                id="fileUpload"
                onChange={(e) => {
                  setFile(e.target.files[0]);
                }}
              />
            </div>
          </div>
          <div>
            <img src="/homeimg.png" alt="" className="max-h-[700px]" />
          </div>
        </div>
      </div>

      <div
        className="pl-[70px] pr-[70px] text-black flex justify-center items-center flex-col gap-[30px] mb-[50px]"
        id="aboutUs"
      >
        <div className="flex gap-8 items-center md:flex-row flex-col">
          <div className="flex flex-col">
            <div className="flex gap-2">
              <p className="text-[50px]">About</p>
              <p className="text-[50px] text-orange-600">Us</p>
            </div>
            <div className="flex flex-col md:text-[20px] text-[15px]">
              <p>
                At VaniSutra, we are dedicated to harnessing the power of
                cutting-edge technology to enhance security and communication in
                critical situations. Our innovative audio analysis system
                combines advanced preprocessing techniques, robust speech
                recognition, and state-of-the-art NLP models to provide
                real-time insights that can make a difference in emergencies
                like kidnappings or interrogations.
              </p>
            </div>
          </div>

          <img src="/aboutus1.png" alt="" className="max-h-[400px]" />
        </div>

        <div className="flex gap-8 items-center md:flex-row flex-col-reverse">
          <img src="/aboutus2.png" alt="" className="max-h-[300px]" />

          <div className="flex flex-col">
            <div className="flex gap-2">
              <p className="text-[50px]">Our</p>
              <p className="text-[50px] text-orange-600">Mission</p>
            </div>
            <div className="flex flex-col md:text-[20px] text-[15px]">
              <p>
                At VaniSutra, we are dedicated to harnessing the power of
                cutting-edge technology to enhance security and communication in
                critical situations. Our innovative audio analysis system
                combines advanced preprocessing techniques, robust speech
                recognition, and state-of-the-art NLP models to provide
                real-time insights that can make a difference in emergencies
                like kidnappings or interrogations.
              </p>
            </div>
          </div>
        </div>
      </div>

      <div
        className="pl-[70px] pr-[70px] text-black flex justify-center items-center flex-col gap-[30px] pb-[50px]"
        id="features"
      >
        <div className="flex text-[50px] font-bold">
          <p>Feat</p>
          <p className="text-orange-600">ures</p>
        </div>

        <div className="flex text-black w-full flex-wrap gap-[80px] justify-center">
          <div className="w-[600px] flex flex-col shadow-md shadow-black p-[10px] rounded-[30px] cursor-pointer transition-transform duration-300 hover:scale-105">
            <div className="h-full flex justify-center items-center">
              <img src="/feature1.png" alt="" />
            </div>

            <p className="text-[#A0674E] md:text-[30px] text-[20px] flex justify-center text-center">
              Language-Agnostic Keyword Spotting
            </p>
            <div className="flex flex-col md:text-[20px] text-[15px] mt-[10px]">
              <p className="text-center">
                The system identifies keywords across multiple languages using
                advanced models like Wav2Vec 2.0 and BERT, ensuring accurate
                keyword detection without relying on language-specific features.
              </p>
            </div>
          </div>

          <div className="w-[600px] flex flex-col shadow-md shadow-black p-[10px] rounded-[30px] cursor-pointer transition-transform duration-300 hover:scale-105">
            <div className="h-full flex justify-center items-center">
              <img src="/feature2.png" alt="" />
            </div>

            <p className="text-[#A0674E] md:text-[30px] text-[20px] flex justify-center text-center">
              Context-Aware Detection
            </p>
            <div className="flex flex-col md:text-[20px] text-[15px] mt-[10px]">
              <p className="text-center">
                The system enhances keyword detection by incorporating
                contextual cues such as time of day, user behavior, and
                location, ensuring more relevant and precise keyword
                identification.
              </p>
            </div>
          </div>

          <div className="w-[600px] flex flex-col shadow-md shadow-black p-[10px] rounded-[30px] cursor-pointer transition-transform duration-300 hover:scale-105">
            <div className="h-full flex justify-center items-center">
              <img src="/feature3.png" alt="" />
            </div>

            <p className="text-[#A0674E] md:text-[30px] text-[20px] flex justify-center text-center">
              Real-Time Alerts and Monitoring
            </p>
            <div className="flex flex-col md:text-[20px] text-[15px] mt-[10px]">
              <p className="text-center">
                Integrated real-time alerts notify users of any mismatches or
                errors in the extracted data, while Flask and Prometheus enable
                ongoing system performance monitoring and visualization for
                efficient operations.
              </p>
            </div>
          </div>

          <div className="w-[600px] flex flex-col shadow-md shadow-black p-[10px] rounded-[30px] cursor-pointer transition-transform duration-300 hover:scale-105">
            <div className="h-full flex justify-center items-center">
              <img src="/feature4.png" alt="" />
            </div>

            <p className="text-[#A0674E] md:text-[30px] text-[20px] flex justify-center text-center">
              Real-Time Speech-to-Text Conversion
            </p>
            <div className="flex flex-col md:text-[20px] text-[10px] mt-[10px]">
              <p className="text-center">
                Audio is converted into text in real-time using tools like
                Google Speech-to-Text or DeepSpeech, allowing for immediate
                transcription and analysis of spoken content.
              </p>
            </div>
          </div>
        </div>
      </div>

      <div
        className="pl-[70px] pr-[70px] text-black flex justify-center items-center flex-col gap-[30px] pb-[50px]"
        id="services"
      >
        <div className="flex text-[50px] font-bold">
          <p>Serv</p>
          <p className="text-orange-600">ices</p>
        </div>

        <div className="flex text-black w-full flex-wrap gap-[80px] justify-center">
          <div className="w-[600px] flex flex-col shadow-md shadow-black p-[10px] rounded-[30px] cursor-pointer transition-transform duration-300 hover:scale-105">
            <div className="h-full flex justify-center items-center">
              <img src="/service1.png" alt="" />
            </div>

            <p className="text-[#A0674E] md:text-[30px] text-[20px] flex justify-center text-center">
              Tailored Speech Recognition Solutions
            </p>
            <div className="flex flex-col md:text-[20px] text-[15px] mt-[10px]">
              <p className="text-center">
                We design and implement customized speech-to-text systems for
                organizations, enabling seamless conversion of spoken language
                into accurate, actionable data tailored to specific industries
                like legal, healthcare, and customer service.
              </p>
            </div>
          </div>

          <div className="w-[600px] flex flex-col shadow-md shadow-black p-[10px] rounded-[30px] cursor-pointer transition-transform duration-300 hover:scale-105">
            <div className="h-full flex justify-center items-center">
              <img src="/service2.png" alt="" />
            </div>

            <p className="text-[#A0674E] md:text-[30px] text-[20px] flex justify-center text-center">
              Keyword Monitoring & Compliance Audits
            </p>
            <div className="flex flex-col md:text-[20px] text-[15px] mt-[10px]">
              <p className="text-center">
                Our service offers real-time keyword monitoring to help
                organizations track and analyze conversations for compliance,
                security, or quality assurance purposes, ensuring adherence to
                industry regulations and internal standards.
              </p>
            </div>
          </div>

          <div className="w-[600px] flex flex-col shadow-md shadow-black p-[10px] rounded-[30px] cursor-pointer transition-transform duration-300 hover:scale-105">
            <div className="h-full flex justify-center items-center">
              <img src="/service3.png" alt="" />
            </div>

            <p className="text-[#A0674E] md:text-[30px] text-[20px] flex justify-center text-center">
              Enterprise-Grade Audio Analytics Integration
            </p>
            <div className="flex flex-col md:text-[20px] text-[15px] mt-[10px]">
              <p className="text-center">
                We integrate advanced audio analytics into your existing
                platforms, providing detailed insights from voice data that
                enhance decision-making and operational efficiency, while
                ensuring compatibility with your existing tools and
                infrastructure.
              </p>
            </div>
          </div>

          <div className="w-[600px] flex flex-col shadow-md shadow-black p-[10px] rounded-[30px] cursor-pointer transition-transform duration-300 hover:scale-105">
            <div className="h-full flex justify-center items-center">
              <img src="/service4.png" alt="" />
            </div>

            <p className="text-[#A0674E] md:text-[30px] text-[20px] flex justify-center text-center">
              24/7 Real-Time Alerting & Support
            </p>
            <div className="flex flex-col md:text-[20px] text-[15px] mt-[10px]">
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

      <div className="bg-[#D0CFCE2E] text-black md:pl-[160px] md:pr-[160px] pl-[10px] pr-[10px] pb-[20px] pt-[20px] overflow-hidden">
        <div className="flex gap-2 md:text-[50px] text-[25px] justify-center mb-[20px] font-bold">
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

        <div className="flex flex-col gap-[20px] w-full">
          <div
            tabIndex={0}
            className="collapse collapse-plus border-base-300 bg-white border text-black"
          >
            <div className="collapse-title text-xl font-medium">
              How will it help us?
            </div>
            <div className="collapse-content">
              <p>
                Our services enhance operational efficiency by automating speech
                recognition and keyword monitoring, ensuring compliance and
                security with real-time alerts. We integrate advanced audio
                analytics for data-driven decision-making and provide 24/7
                support, ensuring your systems are reliable, secure, and always
                operational.
              </p>
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
              <p>
                Our system is utilized by organizations across various
                industries such as healthcare, legal, customer service, and
                security. These sectors rely on accurate speech recognition,
                real-time monitoring, and compliance checks to streamline their
                workflows, enhance decision-making, and ensure regulatory
                adherence.
              </p>
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
              <p>
                The system uses a robust tech stack, including Python for
                backend processing, TensorFlow and PyTorch for machine learning,
                Wav2Vec 2.0 for speech recognition, and BERT for natural
                language processing. We also employ Flask and Prometheus for
                real-time monitoring and alerts, integrated with cloud platforms
                like AWS.
              </p>
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
              <p>
                This system is crucial for automating the processing of large
                volumes of audio data, ensuring compliance with regulatory
                standards, improving operational efficiency, and enabling
                real-time insights. It helps organizations save time and
                resources while enhancing decision-making and ensuring security
                through timely alerts.
              </p>
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
              <p>
                The primary purpose of the system is to convert speech into
                actionable insights through real-time keyword spotting, ensuring
                compliance and security, and providing detailed analytics. It
                serves to optimize workflows, automate tedious tasks, and
                support data-driven decision-making in organizations
                handling audio data.
              </p>
            </div>
          </div>
        </div>
      </div>

      <div className="flex flex-col text-black" id="contactUs">
        <div className="flex p-[100px] gap-[100px] justify-center flex-col-reverse md:flex-row items-center">
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
            <form onSubmit={contactUs}>
              <div className="flex flex-col gap-2">
                <div>
                  <div className="flex gap-2 md:flex-row flex-col">
                    <input
                      type="text"
                      placeholder="Your Name"
                      name="Name"
                      className="input input-bordered"
                      onChange={(e) => {
                        setDetails({ ...details, yourName: e.target.value });
                      }}
                      defaultValue={details.yourName}
                    />
                    <input
                      type="email"
                      name="Email"
                      placeholder="Email"
                      className="input input-bordered"
                      onChange={(e) => {
                        setDetails({ ...details, email: e.target.value });
                      }}
                      defaultValue={details.email}
                    />
                  </div>
                </div>

                <div>
                  <div className="flex gap-2 md:flex-row flex-col">
                    <input
                      type="number"
                      placeholder="Phone no"
                      className="input input-bordered"
                      name="Phone No"
                      onChange={(e) => {
                        setDetails({ ...details, phoneNo: e.target.value });
                      }}
                      defaultValue={details.phoneNo}
                    />
                    <input
                      type="text"
                      placeholder="Gender"
                      name="Gender"
                      className="input input-bordered"
                      onChange={(e) => {
                        setDetails({ ...details, gender: e.target.value });
                      }}
                      defaultValue={details.gender}
                    />
                  </div>
                </div>

                <div>
                  <textarea
                    className="textarea textarea-bordered w-full"
                    name="Message"
                    placeholder="Message"
                    onChange={(e) => {
                      setDetails({ ...details, message: e.target.value });
                    }}
                    defaultValue={details.message}
                  ></textarea>
                </div>

                <div>
                  <button className="bg-black text-orange-600 font-bold h-[60px] w-[300px] rounded-[20px] text-[30px]">
                    Submit
                  </button>
                </div>
              </div>
            </form>
          </div>
        </div>

        <div className="bg-black flex justify-center h-[80px] text-[30px] font-bold text-white w-full items-center text-center">
          All Right Reserved @vanisutra
        </div>
      </div>
    </div>
  );
}
