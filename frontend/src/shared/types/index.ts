// 백엔드(api/service.py) 응답과 1:1 — camelCase 정합.
export type Gate = 'supervised' | 'mahalanobis'
export type Category = '가스' | '미성형' | '정상'

export interface ShotItem {
  idx: number
  groundTruth: string
}

export interface EvidenceItem {
  var: string
  observed: number
  normalMean: number
  normalSd: number
  z: number
}

export interface ZBar {
  var: string
  z: number
  mode: string | null // '가스' | '미성형' | null → 색상 매핑
}

export interface Diagnosis {
  verdict: string
  isAnomaly: boolean
  pValue: number
  alpha: number
  score: number
  gateMode: string
  mode: string | null
  groundTruth: string
  evidence: EvidenceItem[]
  zbars: ZBar[]
  prescription: string | null
}

export interface Trust {
  recallUnsup: number
  recallSup: number
  recallCi: [number, number]
  cwOurs: number
  cwVanilla: number
  cwDiffCi: [number, number]
  fprGuarantee: number
  alpha: number
  rg3Fpr: number
  rg3Recall: number
}
