import type { AgentMessage } from "@mariozechner/pi-agent-core";

type AnthropicContentBlock = {
  type: "text" | "toolUse" | "toolResult";
  text?: string;
  id?: string;
  name?: string;
  toolUseId?: string;
};

/**
 * Strips dangling tool_use blocks from assistant messages when no matching
 * tool_result block exists anywhere in the conversation.
 * This fixes the "tool_use ids found without tool_result blocks" error from Anthropic.
 *
 * Uses a single O(n) pre-pass to build a Set of all tool_result IDs,
 * then checks membership in O(1) — avoiding the previous O(n²) scan-ahead.
 */
function stripDanglingAnthropicToolUses(messages: AgentMessage[]): AgentMessage[] {
  // O(n) pre-pass: collect all tool_result IDs from the entire conversation
  const allToolResultIds = new Set<string>();
  for (const msg of messages) {
    if (!msg || typeof msg !== "object") continue;
    const content = (msg as { content?: AnthropicContentBlock[] }).content;
    if (!Array.isArray(content)) continue;
    for (const block of content) {
      if (block && block.type === "toolResult" && block.toolUseId) {
        allToolResultIds.add(block.toolUseId);
      }
    }
  }

  const result: AgentMessage[] = [];

  for (let i = 0; i < messages.length; i++) {
    const msg = messages[i];
    if (!msg || typeof msg !== "object") {
      result.push(msg);
      continue;
    }

    const msgRole = (msg as { role?: unknown }).role as string | undefined;
    if (msgRole !== "assistant") {
      result.push(msg);
      continue;
    }

    const assistantMsg = msg as {
      content?: AnthropicContentBlock[];
    };

    // Filter out tool_use blocks whose IDs have no matching tool_result anywhere
    const originalContent = Array.isArray(assistantMsg.content) ? assistantMsg.content : [];
    const filteredContent = originalContent.filter((block) => {
      if (!block) return false;
      if (block.type !== "toolUse") return true;
      // O(1) lookup — no scan-ahead needed
      return allToolResultIds.has(block.id || "");
    });

    // If all content would be removed, insert a minimal fallback text block
    if (originalContent.length > 0 && filteredContent.length === 0) {
      result.push({
        ...assistantMsg,
        content: [{ type: "text", text: "[tool calls omitted]" }],
      } as AgentMessage);
    } else {
      result.push({
        ...assistantMsg,
        content: filteredContent,
      } as AgentMessage);
    }
  }

  return result;
}

function validateTurnsWithConsecutiveMerge<TRole extends "assistant" | "user">(params: {
  messages: AgentMessage[];
  role: TRole;
  merge: (
    previous: Extract<AgentMessage, { role: TRole }>,
    current: Extract<AgentMessage, { role: TRole }>,
  ) => Extract<AgentMessage, { role: TRole }>;
}): AgentMessage[] {
  const { messages, role, merge } = params;
  if (!Array.isArray(messages) || messages.length === 0) {
    return messages;
  }

  const result: AgentMessage[] = [];
  let lastRole: string | undefined;

  for (const msg of messages) {
    if (!msg || typeof msg !== "object") {
      result.push(msg);
      continue;
    }

    const msgRole = (msg as { role?: unknown }).role as string | undefined;
    if (!msgRole) {
      result.push(msg);
      continue;
    }

    if (msgRole === lastRole && lastRole === role) {
      const lastMsg = result[result.length - 1];
      const currentMsg = msg as Extract<AgentMessage, { role: TRole }>;

      if (lastMsg && typeof lastMsg === "object") {
        const lastTyped = lastMsg as Extract<AgentMessage, { role: TRole }>;
        result[result.length - 1] = merge(lastTyped, currentMsg);
        continue;
      }
    }

    result.push(msg);
    lastRole = msgRole;
  }

  return result;
}

function mergeConsecutiveAssistantTurns(
  previous: Extract<AgentMessage, { role: "assistant" }>,
  current: Extract<AgentMessage, { role: "assistant" }>,
): Extract<AgentMessage, { role: "assistant" }> {
  const mergedContent = [
    ...(Array.isArray(previous.content) ? previous.content : []),
    ...(Array.isArray(current.content) ? current.content : []),
  ];
  return {
    ...previous,
    content: mergedContent,
    ...(current.usage && { usage: current.usage }),
    ...(current.stopReason && { stopReason: current.stopReason }),
    ...(current.errorMessage && {
      errorMessage: current.errorMessage,
    }),
  };
}

/**
 * Validates and fixes conversation turn sequences for Gemini API.
 * Gemini requires strict alternating user→assistant→tool→user pattern.
 * Merges consecutive assistant messages together.
 */
export function validateGeminiTurns(messages: AgentMessage[]): AgentMessage[] {
  return validateTurnsWithConsecutiveMerge({
    messages,
    role: "assistant",
    merge: mergeConsecutiveAssistantTurns,
  });
}

export function mergeConsecutiveUserTurns(
  previous: Extract<AgentMessage, { role: "user" }>,
  current: Extract<AgentMessage, { role: "user" }>,
): Extract<AgentMessage, { role: "user" }> {
  const mergedContent = [
    ...(Array.isArray(previous.content) ? previous.content : []),
    ...(Array.isArray(current.content) ? current.content : []),
  ];

  return {
    ...current,
    content: mergedContent,
    timestamp: current.timestamp ?? previous.timestamp,
  };
}

/**
 * Validates and fixes conversation turn sequences for Anthropic API.
 * Anthropic requires strict alternating user→assistant pattern.
 * Merges consecutive user messages together.
 * Also strips dangling tool_use blocks that lack corresponding tool_result blocks.
 */
export function validateAnthropicTurns(messages: AgentMessage[]): AgentMessage[] {
  // First, strip dangling tool_use blocks from assistant messages
  const stripped = stripDanglingAnthropicToolUses(messages);

  return validateTurnsWithConsecutiveMerge({
    messages: stripped,
    role: "user",
    merge: mergeConsecutiveUserTurns,
  });
}
