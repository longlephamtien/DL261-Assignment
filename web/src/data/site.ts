import type { AiDisclosure, Course, Group, Institution } from "./types";

export const institution: Institution = {
  university: "Ho Chi Minh City University of Technology (VNU-HCM)",
  faculty: "Faculty of Computer Science and Engineering",
};

export const course: Course = {
  name: "Deep Learning and Its Applications",
  code: "CO3133",
  semester: "Semester 261",
  instructor: "Lê Thành Sách",
};

export const group: Group = {
  id: "18",
  name: "DeepDive",
  repository: "https://github.com/longlephamtien/DL261-Assignment",
  members: [
    {
      name: "Lê Phạm Tiến Long",
      studentId: "2352688",
      role: "Data analysis and delivery: EDA, dataset proposals, repository, website, reports, and submissions",
      github: "https://github.com/longlephamtien",
    },
    {
      name: "Ngô Tiểu Nghi",
      studentId: "2352799",
      role: "Data and training pipeline: loaders, preprocessing, splits, training, fine-tuning, and experiment runs",
      github: "http://github.com/nghingo169",
    },
    {
      name: "Hồ Minh Nhật",
      studentId: "2352858",
      role: "Models and evaluation: architectures, fusion, metrics, ablations, and error analysis",
      github: "https://github.com/henries05",
    },
  ],
};

export const courseAiDisclosure: AiDisclosure = {
  noAiUsed: false,
  summary: null,
  entries: [],
};

export const site = {
  name: "DL261-Assignment",
  title: `${course.code}: ${course.name}`,
  description: `Course project website for ${course.name} (${course.code}), ${course.semester}, ${institution.university}.`,
};
