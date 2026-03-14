import { describe, it, expect, vi, beforeEach } from "vitest";
import type { ExportMetadata } from "mutafitup-common";

// Mock modules before imports
vi.mock("onnxruntime-web/all", () => {
  // Must use a function (not arrow) so it can be called with `new`
  function MockTensor(this: any, type: string, data: any, dims: number[]) {
    this.type = type;
    this.data = data;
    this.dims = dims;
  }

  return {
    InferenceSession: {
      create: vi.fn(),
    },
    Tensor: MockTensor,
  };
});

vi.mock("mutafitup-common", async (importOriginal) => {
  const actual = await importOriginal<typeof import("mutafitup-common")>();
  return {
    ...actual,
    tokenize: vi.fn(),
  };
});

vi.mock("../tokenizer.js", () => ({
  loadTokenizer: vi.fn(),
}));

// Import after mocks are set up
const { MutafitupModel } = await import("../index.js");
const ort = await import("onnxruntime-web/all");
const { loadTokenizer } = await import("../tokenizer.js");
const { tokenize } = await import("mutafitup-common");

const MOCK_METADATA: ExportMetadata = {
  base_checkpoint: "facebook/esm2_t6_8M_UR50D",
  backbone_class: "mutafitup.models.multitask_backbones.esm_backbone.EsmBackbone",
  preprocessing: {
    space_separate: false,
    prefix: null,
    char_replacements: {},
  },
  tasks: {
    meltome: {
      problem_type: "regression",
      level: "per_protein",
      num_outputs: 1,
      output_name: "meltome_logits",
    },
    subloc: {
      problem_type: "classification",
      level: "per_protein",
      num_outputs: 10,
      output_name: "subloc_logits",
    },
  },
  inputs: ["input_ids", "attention_mask"],
  outputs: ["meltome_logits", "subloc_logits"],
  opset_version: 18,
};

describe("MutafitupModel", () => {
  let mockSession: {
    run: ReturnType<typeof vi.fn>;
    release: ReturnType<typeof vi.fn>;
  };

  beforeEach(() => {
    vi.clearAllMocks();

    mockSession = {
      run: vi.fn(),
      release: vi.fn(),
    };

    vi.mocked(loadTokenizer).mockResolvedValue({} as any);
    vi.mocked(ort.InferenceSession.create).mockResolvedValue(mockSession as any);
  });

  describe("load from URL", () => {
    beforeEach(() => {
      // Mock global fetch for metadata loading
      vi.stubGlobal("fetch", vi.fn().mockResolvedValue({
        ok: true,
        json: vi.fn().mockResolvedValue(MOCK_METADATA),
      }));
    });

    it("fetches metadata, loads tokenizer, and creates ONNX session", async () => {
      const model = await MutafitupModel.load("https://example.com/models/my-model");

      expect(fetch).toHaveBeenCalledWith(
        "https://example.com/models/my-model/export_metadata.json",
      );
      expect(loadTokenizer).toHaveBeenCalledWith(
        "https://example.com/models/my-model/tokenizer",
      );
      expect(ort.InferenceSession.create).toHaveBeenCalledWith(
        "https://example.com/models/my-model/model.onnx",
        {
          executionProviders: [
            { name: "webnn", deviceType: "gpu", powerPreference: "high-performance" },
            { name: "webgpu" },
            { name: "wasm" },
          ],
        },
      );

      model.dispose();
    });

    it("strips trailing slash from URL prefix", async () => {
      const model = await MutafitupModel.load("https://example.com/models/my-model/");

      expect(fetch).toHaveBeenCalledWith(
        "https://example.com/models/my-model/export_metadata.json",
      );

      model.dispose();
    });

    it("uses custom execution providers when specified", async () => {
      const model = await MutafitupModel.load("https://example.com/models/my-model", {
        executionProviders: [{ name: "wasm" }],
      });

      expect(ort.InferenceSession.create).toHaveBeenCalledWith(
        "https://example.com/models/my-model/model.onnx",
        {
          executionProviders: [{ name: "wasm" }],
        },
      );

      model.dispose();
    });

    it("throws on fetch failure", async () => {
      vi.mocked(fetch).mockResolvedValue({
        ok: false,
        status: 404,
        statusText: "Not Found",
      } as any);

      await expect(
        MutafitupModel.load("https://example.com/bad-path"),
      ).rejects.toThrow("Failed to fetch export_metadata.json: 404 Not Found");
    });

    it("exposes loaded metadata", async () => {
      const model = await MutafitupModel.load("https://example.com/models/my-model");

      expect(model.metadata).toEqual(MOCK_METADATA);
      expect(model.metadata.tasks.meltome.problem_type).toBe("regression");
      expect(model.metadata.tasks.subloc.num_outputs).toBe(10);

      model.dispose();
    });
  });

  describe("load from object", () => {
    it("loads from pre-fetched metadata and model ArrayBuffer", async () => {
      const modelBuffer = new ArrayBuffer(16);

      const model = await MutafitupModel.load({
        metadata: MOCK_METADATA,
        model: modelBuffer,
        tokenizerSource: "https://example.com/tokenizer",
      });

      expect(loadTokenizer).toHaveBeenCalledWith("https://example.com/tokenizer");
      // ArrayBuffer is wrapped in Uint8Array for onnxruntime-web
      const createCall = vi.mocked(ort.InferenceSession.create).mock.calls[0];
      expect(createCall[0]).toBeInstanceOf(Uint8Array);

      model.dispose();
    });

    it("loads from pre-fetched metadata and model URL string", async () => {
      const model = await MutafitupModel.load({
        metadata: MOCK_METADATA,
        model: "https://example.com/model.onnx",
        tokenizerSource: "https://example.com/tokenizer",
      });

      expect(ort.InferenceSession.create).toHaveBeenCalledWith(
        "https://example.com/model.onnx",
        expect.objectContaining({ executionProviders: expect.any(Array) }),
      );

      model.dispose();
    });
  });

  describe("predict", () => {
    beforeEach(() => {
      vi.stubGlobal("fetch", vi.fn().mockResolvedValue({
        ok: true,
        json: vi.fn().mockResolvedValue(MOCK_METADATA),
      }));
    });

    it("preprocesses, tokenizes, runs session, and returns predictions", async () => {
      const model = await MutafitupModel.load("https://example.com/models/my-model");

      // Mock tokenize to return tensors for 1 sequence of length 5
      vi.mocked(tokenize).mockReturnValue({
        inputIds: BigInt64Array.from([0n, 4n, 5n, 6n, 2n]),
        attentionMask: BigInt64Array.from([1n, 1n, 1n, 1n, 1n]),
        batchSize: 1,
        seqLen: 5,
      });

      // Mock session.run to return task outputs
      mockSession.run.mockResolvedValue({
        meltome_logits: { data: new Float32Array([42.5]) },
        subloc_logits: { data: new Float32Array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]) },
      });

      const predictions = await model.predict(["MAKLV"]);

      // Verify tokenize was called with preprocessed sequences
      expect(tokenize).toHaveBeenCalledWith(
        expect.anything(),
        ["MAKLV"],
      );

      // Verify session.run was called with correct tensor inputs
      expect(mockSession.run).toHaveBeenCalledOnce();
      const feeds = mockSession.run.mock.calls[0][0];
      expect(feeds.input_ids.dims).toEqual([1, 5]);
      expect(feeds.attention_mask.dims).toEqual([1, 5]);

      // Verify output extraction
      expect(predictions.meltome).toBeInstanceOf(Float32Array);
      expect(predictions.meltome[0]).toBeCloseTo(42.5);
      expect(predictions.subloc).toBeInstanceOf(Float32Array);
      expect(predictions.subloc.length).toBe(10);

      model.dispose();
    });

    it("handles batch predictions", async () => {
      const model = await MutafitupModel.load("https://example.com/models/my-model");

      vi.mocked(tokenize).mockReturnValue({
        inputIds: BigInt64Array.from([0n, 4n, 2n, 1n, 0n, 5n, 6n, 2n]),
        attentionMask: BigInt64Array.from([1n, 1n, 1n, 0n, 1n, 1n, 1n, 1n]),
        batchSize: 2,
        seqLen: 4,
      });

      mockSession.run.mockResolvedValue({
        meltome_logits: { data: new Float32Array([42.5, 37.2]) },
        subloc_logits: { data: new Float32Array(20) },
      });

      const predictions = await model.predict(["M", "AKL"]);

      const feeds = mockSession.run.mock.calls[0][0];
      expect(feeds.input_ids.dims).toEqual([2, 4]);

      expect(predictions.meltome.length).toBe(2);
      expect(predictions.subloc.length).toBe(20);

      model.dispose();
    });

    it("chunks sequences when batchSize option is set", async () => {
      const model = await MutafitupModel.load("https://example.com/models/my-model");

      vi.mocked(tokenize).mockImplementation((_tok, seqs) => {
        const n = (seqs as string[]).length;
        return {
          inputIds: new BigInt64Array(n * 3),
          attentionMask: new BigInt64Array(n * 3).fill(1n),
          batchSize: n,
          seqLen: 3,
        };
      });

      mockSession.run
        .mockResolvedValueOnce({
          meltome_logits: { data: new Float32Array([1.0, 2.0]) },
          subloc_logits: { data: new Float32Array(20) },
        })
        .mockResolvedValueOnce({
          meltome_logits: { data: new Float32Array([3.0, 4.0]) },
          subloc_logits: { data: new Float32Array(20) },
        });

      const predictions = await model.predict(
        ["A", "B", "C", "D"],
        { batchSize: 2 },
      );

      expect(mockSession.run).toHaveBeenCalledTimes(2);

      expect(predictions.meltome.length).toBe(4);
      expect(predictions.meltome[0]).toBeCloseTo(1.0);
      expect(predictions.meltome[1]).toBeCloseTo(2.0);
      expect(predictions.meltome[2]).toBeCloseTo(3.0);
      expect(predictions.meltome[3]).toBeCloseTo(4.0);

      expect(predictions.subloc.length).toBe(40);

      model.dispose();
    });

    it("does not chunk when batchSize >= sequence count", async () => {
      const model = await MutafitupModel.load("https://example.com/models/my-model");

      vi.mocked(tokenize).mockReturnValue({
        inputIds: BigInt64Array.from([0n, 4n, 2n, 0n, 5n, 2n]),
        attentionMask: BigInt64Array.from([1n, 1n, 1n, 1n, 1n, 1n]),
        batchSize: 2,
        seqLen: 3,
      });

      mockSession.run.mockResolvedValue({
        meltome_logits: { data: new Float32Array([1.0, 2.0]) },
        subloc_logits: { data: new Float32Array(20) },
      });

      const predictions = await model.predict(
        ["A", "B"],
        { batchSize: 10 },
      );

      expect(mockSession.run).toHaveBeenCalledOnce();
      expect(predictions.meltome.length).toBe(2);

      model.dispose();
    });
  });

  describe("dispose", () => {
    it("releases the ONNX session", async () => {
      vi.stubGlobal("fetch", vi.fn().mockResolvedValue({
        ok: true,
        json: vi.fn().mockResolvedValue(MOCK_METADATA),
      }));

      const model = await MutafitupModel.load("https://example.com/models/my-model");

      model.dispose();

      expect(mockSession.release).toHaveBeenCalledOnce();
    });
  });
});
