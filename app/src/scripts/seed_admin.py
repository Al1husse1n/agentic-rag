"""Database Seeding Script for Initial Root Administrator and RBAC Baseline.

What this script does:
1. Bootstraps baseline permissions and roles (Super Admin, admin, Instructor).
2. Establishes M:N role-permission associations using direct relational queries
   (avoiding async lazy-load greenlet issues).
3. Creates the root administrator account (`admin@academy.edu`) with Argon2id password hashing.
4. Anchors the self-referential audit trail (`created_by = admin.id`, `updated_by = admin.id`)
   to resolve the chicken-and-egg database constraint problem.
5. Ensures idempotency: running this script repeatedly will not create duplicates or fail.

Usage:
    poetry run python -m src.scripts.seed_admin
"""

import asyncio
import logging
import os
import sys
import uuid

from sqlalchemy import select

from src.core.database import AsyncSessionLocal, engine
from src.core.security import hash_password
from src.models.user import (
    Permission,
    Role,
    RolePermission,
    User,
    UserRole,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("seed_admin")

# ---------------------------------------------------------------------------
# Default Configuration (Can be overridden via environment variables)
# ---------------------------------------------------------------------------
DEFAULT_ADMIN_EMAIL = os.getenv("SEED_ADMIN_EMAIL", "admin@academy.edu")
DEFAULT_ADMIN_PASSWORD = os.getenv("SEED_ADMIN_PASSWORD", "AdminPassword123!")
DEFAULT_ADMIN_NAME = os.getenv("SEED_ADMIN_NAME", "System Administrator")

# Baseline permissions required by dashboard modules
PERMISSIONS_DEFINITIONS: dict[str, str] = {
    "documents:read": "View and search knowledge base documents",
    "documents:write": "Upload, update, and manage knowledge base documents",
    "audit:read": "View human-in-the-loop (HITL) audit logs and unresolved inquiries",
    "audit:write": "Annotate, review, and resolve HITL inquiries",
    "users:read": "View system user accounts and assigned roles",
    "users:write": "Create, modify, and deactivate user accounts and permissions",
    "system:manage": "Access and manage core system settings and vector indexes",
}

# Baseline roles and their mapped permissions
ROLES_DEFINITIONS: dict[str, dict[str, object]] = {
    "Super Admin": {
        "description": "Root administrator with full unrestricted access across all systems",
        "permissions": list(PERMISSIONS_DEFINITIONS.keys()),
    },
    "admin": {
        "description": "Standard system administrator with full management privileges",
        "permissions": list(PERMISSIONS_DEFINITIONS.keys()),
    },
    "Instructor": {
        "description": "Academic instructor with access to documents and inquiry reviews",
        "permissions": [
            "documents:read",
            "documents:write",
            "audit:read",
            "audit:write",
        ],
    },
}


async def seed_permissions(session: AsyncSessionLocal) -> dict[str, Permission]:  # type: ignore[valid-type]
    """Idempotently seed all granular permissions."""
    logger.info("Verifying granular permissions...")
    stmt = select(Permission)
    result = await session.execute(stmt)
    existing_perms = {p.name: p for p in result.scalars().all()}

    created_count = 0
    for name, description in PERMISSIONS_DEFINITIONS.items():
        if name not in existing_perms:
            new_perm = Permission(
                id=uuid.uuid4(),
                name=name,
                description=description,
            )
            session.add(new_perm)
            existing_perms[name] = new_perm
            created_count += 1

    if created_count > 0:
        await session.flush()
        logger.info("Created %d new granular permissions.", created_count)
    else:
        logger.info("All granular permissions are already up to date.")

    return existing_perms


async def seed_roles(
    session: AsyncSessionLocal,  # type: ignore[valid-type]
    perms_map: dict[str, Permission],
) -> dict[str, Role]:
    """Idempotently seed roles and link their role permissions."""
    logger.info("Verifying system roles and permission mappings...")
    stmt = select(Role)
    result = await session.execute(stmt)
    existing_roles = {r.name: r for r in result.scalars().all()}

    # Ensure all required roles exist
    for role_name, config in ROLES_DEFINITIONS.items():
        description = str(config["description"])
        if role_name not in existing_roles:
            role = Role(
                id=uuid.uuid4(),
                name=role_name,
                description=description,
            )
            session.add(role)
            await session.flush()
            existing_roles[role_name] = role
            logger.info("Created role '%s'.", role_name)

    # Fetch existing role-permission links to prevent duplicate associations
    rp_stmt = select(RolePermission)
    rp_result = await session.execute(rp_stmt)
    existing_links = {
        (rp.role_id, rp.permission_id) for rp in rp_result.scalars().all()
    }

    # Add missing links
    links_created = 0
    for role_name, config in ROLES_DEFINITIONS.items():
        role = existing_roles[role_name]
        target_perm_names = list(config["permissions"])  # type: ignore[arg-type]

        for perm_name in target_perm_names:
            perm = perms_map[perm_name]
            link_key = (role.id, perm.id)
            if link_key not in existing_links:
                session.add(
                    RolePermission(
                        id=uuid.uuid4(),
                        role_id=role.id,
                        permission_id=perm.id,
                    )
                )
                existing_links.add(link_key)
                links_created += 1

    if links_created > 0:
        await session.flush()
        logger.info("Created %d role-permission associations.", links_created)
    else:
        logger.info("All role-permission mappings are already up to date.")

    return existing_roles


async def seed_root_admin(
    session: AsyncSessionLocal,  # type: ignore[valid-type]
    roles_map: dict[str, Role],
) -> User:
    """Idempotently seed the root administrator user and attach admin roles."""
    logger.info("Checking for existing root administrator '%s'...", DEFAULT_ADMIN_EMAIL)
    stmt = select(User).where(User.email_address == DEFAULT_ADMIN_EMAIL)
    result = await session.execute(stmt)
    admin_user = result.scalars().first()

    if admin_user is None:
        logger.info("Creating initial root administrator...")
        admin_id = uuid.uuid4()
        hashed_pw = hash_password(DEFAULT_ADMIN_PASSWORD)

        # Anchoring self-referential audit fields:
        # Since this is the initial user, created_by and updated_by reference admin_id.
        admin_user = User(
            id=admin_id,
            email_address=DEFAULT_ADMIN_EMAIL,
            hashed_password=hashed_pw,
            full_name=DEFAULT_ADMIN_NAME,
            is_active=True,
            created_by=admin_id,
            updated_by=admin_id,
        )
        session.add(admin_user)
        await session.flush()
        logger.info("Root administrator record created with ID: %s", admin_id)
    else:
        logger.info("Root administrator '%s' already exists (ID: %s).", DEFAULT_ADMIN_EMAIL, admin_user.id)
        # Ensure audit fields point to a valid anchor if they were previously null
        if admin_user.created_by is None:
            admin_user.created_by = admin_user.id
        if admin_user.updated_by is None:
            admin_user.updated_by = admin_user.id
        await session.flush()

    # Query existing UserRole associations for the admin
    ur_stmt = select(UserRole).where(UserRole.user_id == admin_user.id)
    ur_result = await session.execute(ur_stmt)
    existing_user_role_ids = {ur.role_id for ur in ur_result.scalars().all()}

    target_roles = ["Super Admin", "admin"]
    roles_assigned = 0
    for role_name in target_roles:
        if role_name in roles_map:
            role = roles_map[role_name]
            if role.id not in existing_user_role_ids:
                session.add(
                    UserRole(
                        id=uuid.uuid4(),
                        user_id=admin_user.id,
                        role_id=role.id,
                    )
                )
                existing_user_role_ids.add(role.id)
                roles_assigned += 1
                logger.info("Assigned role '%s' to root administrator.", role_name)

    if roles_assigned > 0:
        await session.flush()

    return admin_user


async def main() -> None:
    """Execute the end-to-end database seeding process."""
    logger.info("==================================================")
    logger.info("🚀 Starting Database Seeding & RBAC Initialization")
    logger.info("==================================================")

    async with AsyncSessionLocal() as session:
        try:
            # 1. Seed granular permissions
            perms_map = await seed_permissions(session)

            # 2. Seed roles and associate permissions
            roles_map = await seed_roles(session, perms_map)

            # 3. Seed root administrator
            admin = await seed_root_admin(session, roles_map)

            # 4. Commit all transaction changes
            await session.commit()

            logger.info("==================================================")
            logger.info("✅ Database Seeding Completed Successfully!")
            logger.info("--------------------------------------------------")
            logger.info("Root Admin Email:    %s", admin.email_address)
            logger.info("Root Admin Name:     %s", admin.full_name)
            logger.info("Root Admin Password: [CONFIGURED IN SEED SCRIPT / ENV]")
            logger.info("Active Roles:        %s", ", ".join(ROLES_DEFINITIONS.keys()))
            logger.info("Total Permissions:   %d", len(PERMISSIONS_DEFINITIONS))
            logger.info("==================================================")

        except Exception as exc:
            await session.rollback()
            logger.exception("❌ Database seeding failed: %s", exc)
            sys.exit(1)

    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
