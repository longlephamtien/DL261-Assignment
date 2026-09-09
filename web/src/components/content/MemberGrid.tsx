import { SquareArrowOutUpRight, User } from "lucide-react";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import { Card, CardContent } from "@/components/ui/card";
import { PendingBadge } from "@/components/content/PendingBadge";
import { displayValue, githubAvatar } from "@/lib/content";
import type { Member } from "@/data/types";

function MemberCard({ member }: { member: Member }) {
  const name = displayValue(member.name);
  const studentId = displayValue(member.studentId);
  const role = displayValue(member.role);
  const avatar = githubAvatar(member.github);

  return (
    <Card className="h-full">
      <CardContent className="flex h-full flex-col gap-3">
        <div className="flex items-center gap-3">
          <Avatar className="size-9 rounded-lg after:rounded-lg after:mix-blend-normal dark:after:mix-blend-normal">
            {avatar ? (
              <img
                src={avatar}
                alt=""
                loading="lazy"
                className="aspect-square size-full rounded-lg object-cover"
              />
            ) : (
              <AvatarFallback className="rounded-lg">
                <User className="size-4" />
              </AvatarFallback>
            )}
          </Avatar>
          <div className="min-w-0">
            <div className="h-card truncate">{name ?? "Name pending"}</div>
            <div className="text-xs text-muted-foreground">{studentId ?? "ID pending"}</div>
          </div>
        </div>

        <div className="grow">
          {role ? <p className="text-sm text-muted-foreground">{role}</p> : <PendingBadge label="Role pending" />}
        </div>

        {member.github ? (
          <a
            href={member.github}
            target="_blank"
            rel="noreferrer"
            className="inline-flex w-fit items-center gap-1.5 text-sm font-medium text-primary hover:underline"
          >
            GitHub
            <SquareArrowOutUpRight className="size-3.5" />
          </a>
        ) : (
          <PendingBadge label="No profile link" />
        )}
      </CardContent>
    </Card>
  );
}

export function MemberGrid({ members }: { members: Member[] }) {
  if (members.length === 0) {
    return <PendingBadge label="No members listed" />;
  }

  return (
    <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
      {members.map((member, index) => (
        <MemberCard key={index} member={member} />
      ))}
    </div>
  );
}
