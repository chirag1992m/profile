import * as React from 'react'
import { GiCompass } from 'react-icons/gi'
import LoadingIcons from 'react-loading-icons'

import Button from '../Button'
import { TitleBar } from '../TitleBar'

const ContentContainer: React.FC<React.HTMLProps<HTMLDivElement>> = (props) => {
    return (
        <div
            className="mx-auto w-full max-w-3xl px-4 py-4 pb-10 md:px-8"
            {...props}
        />
    )
}

interface DetailContainerProps {
    children: React.ReactNode
}

const Container = React.forwardRef<HTMLDivElement, DetailContainerProps>(
    (props, ref) => {
        return (
            <div
                ref={ref}
                id="main"
                className="flex max-h-screen w-full flex-col overflow-y-auto bg-white dark:bg-black"
                {...props}
            />
        )
    }
)

const Header: React.FC<React.HTMLProps<HTMLDivElement>> = (props) => {
    return <div className="relative w-[560px] h-[375px]" {...props} />
}

interface TitleProps {
    children: React.ReactNode
}

const Title = React.forwardRef<HTMLHeadingElement, TitleProps>((props, ref) => {
    return (
        <h1
            ref={ref}
            className="text-primary font-sans text-3xl font-bold xl:text-3xl absolute bottom-[-30px] left-2.5 bg-white border-2 border-black rounded-lg p-2.5"
            {...props}
        />
    )
})

const Loading: React.FC = () => {
    return (
        <Container>
            <div className="flex flex-1 flex-col items-center justify-center">
                <LoadingIcons.Circles />
            </div>
        </Container>
    )
}

const Null: React.FC = () => {
    return (
        <Container>
            <TitleBar title="Not found" />
            <div className="flex flex-1 flex-col items-center justify-center space-y-6 px-8 text-center lg:px-16">
                <GiCompass className="text-secondary" size={32} />
                <div className="flex flex-col space-y-1">
                    <p className="text-primary font-semibold">
                        What you seek does not exist.
                    </p>
                    <p className="text-tertiary">
                        Maybe this link is broken. Maybe something was deleted,
                        or moved. In any case, there&apos;s nothing to see
                        here...
                    </p>
                </div>
                <Button href="/">Go home</Button>
            </div>
        </Container>
    )
}

export const Detail = {
    Container,
    ContentContainer,
    Header,
    Title,
    Loading,
    Null,
}
