'use client'
import * as React from 'react'

import { GlobalNavigationContext } from './GlobalNavigationContext'

export function ContextProviders({
    children,
}: {
    children: React.ReactNode
}): React.ReactElement {
    const initialState = {
        isOpen: false,
        setIsOpen,
    }

    const [state, setState] = React.useState(initialState)

    function setIsOpen(isOpen: boolean): void {
        setState({ ...state, isOpen })
    }

    return (
        <>
            <GlobalNavigationContext.Provider value={state}>
                {children}
            </GlobalNavigationContext.Provider>
        </>
    )
}