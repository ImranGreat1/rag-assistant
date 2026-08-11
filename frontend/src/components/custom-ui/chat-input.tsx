import { ArrowUpIcon } from "lucide-react";
import { 
    InputGroup, 
    InputGroupAddon, 
    InputGroupInput,
    InputGroupButton
} from "@/components/ui/input-group";
import { Search } from "lucide-react";


export function ChatInput({ onChange, onClick }: any) 
{
    return (
        <InputGroup className="max-w-xl">
            <InputGroupInput onChange={onChange} placeholder="Search..." />
            <InputGroupAddon>
                <Search />
            </InputGroupAddon>
            <InputGroupButton onClick={onClick}>
                <ArrowUpIcon className="outline" />
            </InputGroupButton>
        </InputGroup>
    )
}